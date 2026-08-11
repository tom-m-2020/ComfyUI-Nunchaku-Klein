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
    "nunchaku_klein_text_ref_direct_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_text_ref_direct_test.models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
    TEXT_REF_BALANCE_DIRECT_OPTION,
)
from nunchaku_klein_text_ref_direct_test.nodes.enhancer.common import (
    REFERENCE_CATEGORY,
)
from nunchaku_klein_text_ref_direct_test.nodes.enhancer.ref_latent_weight_direct import (
    KleinRefLatentWeightKVCallback,
)
from nunchaku_klein_text_ref_direct_test.nodes.enhancer.text_ref_balance_direct import (
    KleinTextRefBalanceKVCallback,
)


class Result:
    def __init__(self, sample):
        self.sample = sample


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offload = False

    def forward(self, *, hidden_states, **kwargs):
        return Result(hidden_states.clone())


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


def metadata(block_type="double", refs=(2, 5, 1)):
    text = 3
    generated = 4
    logical_image = generated + sum(refs)
    if block_type == "double":
        padded_text = 8
        padded_image = 16
    else:
        padded_text = text
        padded_image = 16
    return SimpleNamespace(
        block_type=block_type,
        block_index=1,
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
    return query, query.clone().add_(1000), query.clone().add_(2000)


def ranges(info):
    image_start = (
        info.padded_text_token_count
        if info.block_type == "double"
        else info.text_token_count
    )
    generated = (image_start, image_start + info.generated_token_count)
    references = []
    position = generated[1]
    for count in info.reference_token_counts:
        references.append((position, position + count))
        position += count
    return (0, info.text_token_count), generated, references


class TextRefBalanceDirectTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS[
            "NunchakuKleinTextRefBalanceDirectKV"
        ]()
        self.conditioning = [[torch.ones((1, 3, 4)), {"marker": "kept"}]]

    def test_registration_ui_clone_and_conditioning_identity(self):
        cls = type(self.node)
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL", "CONDITIONING"))
        self.assertEqual(cls.FUNCTION, "balance_streams")
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[
                "NunchakuKleinTextRefBalanceDirectKV"
            ],
            "Nunchaku FLUX.2 Klein Text/Ref Balance (Direct K/V)",
        )
        inputs = cls.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["model", "conditioning", "balance"])
        self.assertEqual(list(inputs["optional"]), ["debug"])
        self.assertEqual(
            inputs["required"]["balance"],
            ("FLOAT", {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001}),
        )
        self.assertEqual(inputs["optional"]["debug"], ("BOOLEAN", {"default": False}))

        adapter = make_adapter()
        source = FakeModelPatcher(adapter)
        first = lambda *args: None
        source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION] = (
            Flux2AttentionCallbacks((first,), ())
        )
        branch, returned = self.node.balance_streams(
            source, self.conditioning, 0.75, False
        )
        self.assertIs(returned, self.conditioning)
        self.assertIsNot(branch, source)
        self.assertNotIn(
            TEXT_REF_BALANCE_DIRECT_OPTION,
            source.model_options["transformer_options"],
        )
        callbacks = branch.model_options["transformer_options"][
            ATTENTION_CALLBACKS_OPTION
        ].pre_attention_callbacks
        self.assertEqual(callbacks[0], first)
        self.assertEqual(callbacks[-1], KleinTextRefBalanceKVCallback(1.0, 0.5, False))
        with self.assertRaises(FrozenInstanceError):
            callbacks[-1].text_scale = 0.0

    def assert_scaled(self, balance, text_scale, reference_scale):
        for block_type in ("double", "single"):
            with self.subTest(balance=balance, block_type=block_type):
                info = metadata(block_type)
                q, k, v = tensors(info)
                q_before, k_before, v_before = (
                    tensor.clone() for tensor in (q, k, v)
                )
                callback = KleinTextRefBalanceKVCallback(
                    text_scale, reference_scale
                )
                self.assertIsNone(callback(q, k, v, info))
                text, generated, references = ranges(info)
                self.assertTrue(torch.equal(q, q_before))
                self.assertTrue(
                    torch.equal(
                        k[:, :, text[0]:text[1]],
                        k_before[:, :, text[0]:text[1]] * text_scale,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        v[:, :, text[0]:text[1]],
                        v_before[:, :, text[0]:text[1]] * text_scale,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        k[:, :, generated[0]:generated[1]],
                        k_before[:, :, generated[0]:generated[1]],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        v[:, :, generated[0]:generated[1]],
                        v_before[:, :, generated[0]:generated[1]],
                    )
                )
                for start, end in references:
                    self.assertTrue(
                        torch.equal(
                            k[:, :, start:end], k_before[:, :, start:end] * reference_scale
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            v[:, :, start:end], v_before[:, :, start:end] * reference_scale
                        )
                    )
                logical = torch.zeros(k.shape[2], dtype=torch.bool)
                logical[text[0]:text[1]] = True
                logical[generated[0]:generated[1]] = True
                for start, end in references:
                    logical[start:end] = True
                self.assertTrue(torch.equal(k[:, :, ~logical], k_before[:, :, ~logical]))
                self.assertTrue(torch.equal(v[:, :, ~logical], v_before[:, :, ~logical]))

    def test_piecewise_endpoints_and_intermediate_values(self):
        for balance, text_scale, reference_scale in (
            (0.0, 0.0, 1.0),
            (0.25, 0.5, 1.0),
            (0.5, 1.0, 1.0),
            (0.75, 1.0, 0.5),
            (1.0, 1.0, 0.0),
        ):
            self.assert_scaled(balance, text_scale, reference_scale)

    def test_neutral_is_bitwise_noop(self):
        for block_type in ("double", "single"):
            info = metadata(block_type)
            q, k, v = tensors(info)
            snapshots = tuple(tensor.clone() for tensor in (q, k, v))
            KleinTextRefBalanceKVCallback(1.0, 1.0)(q, k, v, info)
            for current, before in zip((q, k, v), snapshots, strict=True):
                self.assertTrue(torch.equal(current, before))

    def test_zero_reference_branch_still_scales_text(self):
        for block_type in ("double", "single"):
            info = metadata(block_type, refs=())
            q, k, v = tensors(info)
            q_before, k_before, v_before = (
                tensor.clone() for tensor in (q, k, v)
            )
            KleinTextRefBalanceKVCallback(0.5, 1.0)(q, k, v, info)
            text, _, references = ranges(info)
            self.assertEqual(references, [])
            self.assertTrue(torch.equal(q, q_before))
            self.assertTrue(
                torch.equal(k[:, :, :text[1]], k_before[:, :, :text[1]] * 0.5)
            )
            self.assertTrue(
                torch.equal(v[:, :, :text[1]], v_before[:, :, :text[1]] * 0.5)
            )
            self.assertTrue(torch.equal(k[:, :, text[1]:], k_before[:, :, text[1]:]))
            self.assertTrue(torch.equal(v[:, :, text[1]:], v_before[:, :, text[1]:]))

    def test_inconsistent_metadata_fails_before_mutation(self):
        info = metadata("single")
        info.logical_image_token_count += 1
        q, k, v = tensors(metadata("single"))
        snapshots = tuple(tensor.clone() for tensor in (q, k, v))
        with self.assertRaises(ValueError):
            KleinTextRefBalanceKVCallback(0.0, 0.5)(q, k, v, info)
        for current, before in zip((q, k, v), snapshots, strict=True):
            self.assertTrue(torch.equal(current, before))

    def test_direct_ref_weight_stacks_multiplicatively_in_order(self):
        info = metadata("single", refs=(2, 5))
        q, k, v = tensors(info)
        q_before, k_before, v_before = (tensor.clone() for tensor in (q, k, v))
        balance = KleinTextRefBalanceKVCallback(1.0, 0.5)
        selected = KleinRefLatentWeightKVCallback(1, 0.5)
        balance(q, k, v, info)
        selected(q, k, v, info)
        text, generated, references = ranges(info)
        self.assertTrue(torch.equal(q, q_before))
        self.assertTrue(torch.equal(k[:, :, references[0][0]:references[0][1]], k_before[:, :, references[0][0]:references[0][1]] * 0.5))
        self.assertTrue(torch.equal(v[:, :, references[0][0]:references[0][1]], v_before[:, :, references[0][0]:references[0][1]] * 0.5))
        self.assertTrue(torch.equal(k[:, :, references[1][0]:references[1][1]], k_before[:, :, references[1][0]:references[1][1]] * 0.25))
        self.assertTrue(torch.equal(v[:, :, references[1][0]:references[1][1]], v_before[:, :, references[1][0]:references[1][1]] * 0.25))
        self.assertTrue(torch.equal(k[:, :, generated[0]:generated[1]], k_before[:, :, generated[0]:generated[1]]))

        adapter = make_adapter()
        source = FakeModelPatcher(adapter)
        balance_branch = self.node.balance_streams(source, self.conditioning, 0.75)[0]
        ref_node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeightDirectKV"]()
        stacked = ref_node.execute(balance_branch, 1, 0.5)[0]
        callbacks = stacked.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION].pre_attention_callbacks
        self.assertIsInstance(callbacks[0], KleinTextRefBalanceKVCallback)
        self.assertIsInstance(callbacks[1], KleinRefLatentWeightKVCallback)

    def test_prediction_space_mixing_is_rejected_in_both_graph_orders(self):
        adapter = make_adapter()
        source = FakeModelPatcher(adapter)
        prediction_balance = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinTextRefBalance"]()
        prediction_ref = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeight"]()
        for prediction_branch in (
            prediction_balance.balance_streams(source, self.conditioning, 0.5)[0],
            prediction_ref.execute(source, 0, 1.0)[0],
        ):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                self.node.balance_streams(prediction_branch, self.conditioning, 0.5)

        direct = self.node.balance_streams(source, self.conditioning, 0.5)[0]
        reverse_branches = (
            prediction_balance.balance_streams(direct, self.conditioning, 0.5)[0],
            prediction_ref.execute(direct, 0, 1.0)[0],
        )
        reference = torch.ones((1, 2, 1, 1))
        for branch in reverse_branches:
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                adapter(
                    torch.zeros((1, 2, 1, 1)),
                    torch.ones((1,)),
                    torch.zeros((1, 1, 4)),
                    transformer_options=branch.model_options["transformer_options"],
                    ref_latents=[reference],
                    ref_latents_method="index",
                )


if __name__ == "__main__":
    unittest.main()
