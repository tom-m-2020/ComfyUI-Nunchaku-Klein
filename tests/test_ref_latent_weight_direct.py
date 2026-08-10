import copy
from dataclasses import FrozenInstanceError
import importlib.util
import pathlib
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_ref_weight_direct_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_ref_weight_direct_test.models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
)
from nunchaku_klein_ref_weight_direct_test.nodes.enhancer.common import (
    REFERENCE_CATEGORY,
)
from nunchaku_klein_ref_weight_direct_test.nodes.enhancer.ref_latent_weight_direct import (
    KleinRefLatentWeightKVCallback,
)


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offload = False


class FakeBaseModel:
    def __init__(self, adapter):
        self.diffusion_model = adapter


class FakeModelPatcher:
    def __init__(self, adapter):
        self.model = FakeBaseModel(adapter)
        self.model_options = {"transformer_options": {}}
        self.size = 123

    def clone(self):
        branch = FakeModelPatcher(self.model.diffusion_model)
        branch.model = self.model
        branch.model_options = copy.deepcopy(self.model_options)
        branch.size = self.size
        return branch


def make_adapter():
    return NunchakuFlux2KleinAdapter(
        FakeTransformer(),
        in_channels=2,
        context_dim=4,
        patch_size=1,
        axes_dim=(1, 1, 1, 1),
        dtype=torch.float32,
    )


def metadata(block_type="double", refs=(2, 3, 1)):
    text = 3
    generated = 4
    logical_image = generated + sum(refs)
    if block_type == "double":
        padded_text = 8
        padded_image = 12
    else:
        padded_text = text
        padded_image = 13
    return SimpleNamespace(
        block_type=block_type,
        block_index=2,
        text_token_count=text,
        generated_token_count=generated,
        reference_token_counts=refs,
        logical_image_token_count=logical_image,
        padded_text_token_count=padded_text,
        padded_image_token_count=padded_image,
        packed_sequence_length=padded_text + padded_image,
        batch_size=1,
        head_count=2,
        head_dimension=2,
    )


def tensors(info):
    shape = (1, 2, info.packed_sequence_length, 2)
    query = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    key = query.clone().add_(1000)
    value = query.clone().add_(2000)
    return query, key, value


def selected_range(info, reference_index):
    image_start = (
        info.padded_text_token_count
        if info.block_type == "double"
        else info.text_token_count
    )
    start = (
        image_start
        + info.generated_token_count
        + sum(info.reference_token_counts[:reference_index])
    )
    return start, start + info.reference_token_counts[reference_index]


class RefLatentWeightDirectTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS[
            "NunchakuKleinRefLatentWeightDirectKV"
        ]()

    def test_registration_ui_and_immutable_branch_callback(self):
        cls = type(self.node)
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL",))
        self.assertEqual(cls.FUNCTION, "execute")
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[
                "NunchakuKleinRefLatentWeightDirectKV"
            ],
            "Nunchaku FLUX.2 Klein Ref Latent Weight (Direct K/V)",
        )
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "reference_index", "weight"])
        self.assertEqual(
            required["reference_index"],
            ("INT", {"default": 0, "min": 0, "max": 7}),
        )
        self.assertEqual(
            required["weight"],
            ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
        )

        adapter = make_adapter()
        source = FakeModelPatcher(adapter)
        pre = lambda *args: None
        post = lambda *args: None
        source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION] = (
            Flux2AttentionCallbacks((pre,), (post,))
        )
        branch = self.node.execute(source, 1, 0.5)[0]
        self.assertIsNot(branch, source)
        source_callbacks = source.model_options["transformer_options"][
            ATTENTION_CALLBACKS_OPTION
        ]
        branch_callbacks = branch.model_options["transformer_options"][
            ATTENTION_CALLBACKS_OPTION
        ]
        self.assertEqual(source_callbacks.pre_attention_callbacks, (pre,))
        self.assertEqual(branch_callbacks.pre_attention_callbacks[0], pre)
        self.assertEqual(branch_callbacks.post_attention_callbacks, (post,))
        direct = branch_callbacks.pre_attention_callbacks[-1]
        self.assertEqual(direct, KleinRefLatentWeightKVCallback(1, 0.5))
        with self.assertRaises(FrozenInstanceError):
            direct.weight = 2.0

    def test_weight_one_is_bitwise_noop_for_both_stream_types(self):
        for block_type in ("double", "single"):
            with self.subTest(block_type=block_type):
                info = metadata(block_type)
                q, k, v = tensors(info)
                snapshots = tuple(tensor.clone() for tensor in (q, k, v))
                result = KleinRefLatentWeightKVCallback(1, 1.0)(q, k, v, info)
                self.assertIsNone(result)
                for current, before in zip((q, k, v), snapshots, strict=True):
                    self.assertTrue(torch.equal(current, before))
                    self.assertEqual(current.shape, before.shape)
                    self.assertEqual(current.dtype, before.dtype)
                    self.assertEqual(current.device, before.device)

    def test_zero_scales_only_selected_reference_key_and_value(self):
        for block_type in ("double", "single"):
            with self.subTest(block_type=block_type):
                info = metadata(block_type)
                q, k, v = tensors(info)
                q_before, k_before, v_before = (tensor.clone() for tensor in (q, k, v))
                start, end = selected_range(info, 1)
                KleinRefLatentWeightKVCallback(1, 0.0)(q, k, v, info)
                self.assertTrue(torch.equal(q, q_before))
                self.assertEqual(torch.count_nonzero(k[:, :, start:end]), 0)
                self.assertEqual(torch.count_nonzero(v[:, :, start:end]), 0)
                self.assertTrue(torch.equal(k[:, :, :start], k_before[:, :, :start]))
                self.assertTrue(torch.equal(k[:, :, end:], k_before[:, :, end:]))
                self.assertTrue(torch.equal(v[:, :, :start], v_before[:, :, :start]))
                self.assertTrue(torch.equal(v[:, :, end:], v_before[:, :, end:]))

    def test_intermediate_and_extrapolated_weights_resolve_heterogeneous_ranges(self):
        for reference_index, weight in ((0, 0.25), (2, 2.0)):
            for block_type in ("double", "single"):
                with self.subTest(
                    block_type=block_type,
                    reference_index=reference_index,
                    weight=weight,
                ):
                    info = metadata(block_type, refs=(2, 5, 1))
                    q, k, v = tensors(info)
                    q_before, k_before, v_before = (
                        tensor.clone() for tensor in (q, k, v)
                    )
                    start, end = selected_range(info, reference_index)
                    KleinRefLatentWeightKVCallback(reference_index, weight)(
                        q, k, v, info
                    )
                    self.assertTrue(torch.equal(q, q_before))
                    self.assertTrue(
                        torch.equal(
                            k[:, :, start:end], k_before[:, :, start:end] * weight
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            v[:, :, start:end], v_before[:, :, start:end] * weight
                        )
                    )
                    mask = torch.ones(k.shape[2], dtype=torch.bool)
                    mask[start:end] = False
                    self.assertTrue(torch.equal(k[:, :, mask], k_before[:, :, mask]))
                    self.assertTrue(torch.equal(v[:, :, mask], v_before[:, :, mask]))

    def test_invalid_index_and_inconsistent_metadata_fail_before_mutation(self):
        cases = []
        valid = metadata("single")
        cases.append((KleinRefLatentWeightKVCallback(3, 0.0), valid, IndexError))
        inconsistent_image = copy.copy(valid)
        inconsistent_image.logical_image_token_count += 1
        cases.append(
            (KleinRefLatentWeightKVCallback(0, 0.0), inconsistent_image, ValueError)
        )
        inconsistent_packed = copy.copy(valid)
        inconsistent_packed.packed_sequence_length -= 1
        cases.append(
            (KleinRefLatentWeightKVCallback(0, 0.0), inconsistent_packed, ValueError)
        )
        for callback, info, error in cases:
            with self.subTest(error=error):
                q, k, v = tensors(valid)
                snapshots = tuple(tensor.clone() for tensor in (q, k, v))
                with self.assertRaises(error):
                    callback(q, k, v, info)
                for current, before in zip((q, k, v), snapshots, strict=True):
                    self.assertTrue(torch.equal(current, before))


if __name__ == "__main__":
    unittest.main()
