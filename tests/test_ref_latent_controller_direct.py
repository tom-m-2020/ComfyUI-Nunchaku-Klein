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
    "nunchaku_klein_ref_controller_direct_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_ref_controller_direct_test.models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
)
from nunchaku_klein_ref_controller_direct_test.nodes.enhancer.common import REFERENCE_CATEGORY
from nunchaku_klein_ref_controller_direct_test.nodes.enhancer.ref_latent_controller_direct import (
    KleinRefLatentControllerKVCallback,
    SPATIAL_FADE_MODES,
)
from nunchaku_klein_ref_controller_direct_test.nodes.enhancer.ref_latent_weight_direct import (
    KleinRefLatentWeightKVCallback,
)
from nunchaku_klein_ref_controller_direct_test.nodes.enhancer.text_ref_balance_direct import (
    KleinTextRefBalanceKVCallback,
)


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offload = False


class FakeModelPatcher:
    def __init__(self, adapter):
        self.model = SimpleNamespace(diffusion_model=adapter)
        self.model_options = {"transformer_options": {}}
        self.size = 1

    def clone(self):
        branch = FakeModelPatcher(self.model.diffusion_model)
        branch.model = self.model
        branch.model_options = copy.deepcopy(self.model_options)
        return branch


def make_adapter():
    return NunchakuFlux2KleinAdapter(
        FakeTransformer(), in_channels=2, context_dim=4, patch_size=1,
        axes_dim=(1, 1, 1, 1), dtype=torch.float32,
    )


def metadata(block_type="double", refs=(6, 4), shapes=((2, 3), (1, 4))):
    text, generated = 3, 5
    if block_type == "double":
        padded_text, padded_image = 8, 16
    else:
        padded_text, padded_image = text, 21
    return SimpleNamespace(
        block_type=block_type, block_index=4, text_token_count=text,
        generated_token_count=generated, reference_token_counts=refs,
        reference_spatial_shapes=shapes,
        logical_image_token_count=generated + sum(refs),
        padded_text_token_count=padded_text, padded_image_token_count=padded_image,
        packed_sequence_length=padded_text + padded_image,
        batch_size=1, head_count=2, head_dimension=2,
    )


def tensors(info):
    shape = (1, 2, info.packed_sequence_length, 2)
    q = torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + 1
    return q, q.clone() + 1000, q.clone() + 2000


def ref_range(info, index):
    image_start = info.padded_text_token_count if info.block_type == "double" else info.text_token_count
    start = image_start + info.generated_token_count + sum(info.reference_token_counts[:index])
    return start, start + info.reference_token_counts[index]


class RefLatentControllerDirectTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentControllerDirectKV"]()
        self.conditioning = [[torch.ones((1, 3, 4)), {"marker": "kept"}]]

    def test_registration_exact_ui_clone_and_immutability(self):
        cls = type(self.node)
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL", "CONDITIONING"))
        self.assertEqual(cls.FUNCTION, "control")
        self.assertEqual(PACKAGE.NODE_DISPLAY_NAME_MAPPINGS["NunchakuKleinRefLatentControllerDirectKV"], "Nunchaku FLUX.2 Klein Ref Latent Controller (Direct K/V)")
        inputs = cls.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["model", "conditioning", "strength", "reference_index"])
        self.assertEqual(list(inputs["optional"]), ["spatial_fade", "spatial_fade_strength", "debug"])
        self.assertEqual(inputs["required"]["strength"], ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05}))
        self.assertEqual(inputs["optional"]["spatial_fade"][0], list(SPATIAL_FADE_MODES))

        source = FakeModelPatcher(make_adapter())
        first = lambda *args: None
        source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION] = Flux2AttentionCallbacks((first,), ())
        branch, conditioning = self.node.control(source, self.conditioning, 0.5, 1, "none", 0.5, False)
        self.assertIs(conditioning, self.conditioning)
        self.assertIsNot(branch, source)
        self.assertEqual(len(source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION].pre_attention_callbacks), 1)
        callback = branch.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION].pre_attention_callbacks[-1]
        self.assertEqual(callback, KleinRefLatentControllerKVCallback(0.5, 1, "none", 0.5, False))
        with self.assertRaises(FrozenInstanceError):
            callback.strength = 2.0

    def test_scalar_selected_reference_only_for_both_streams(self):
        for block_type in ("double", "single"):
            info = metadata(block_type)
            q, k, v = tensors(info)
            qb, kb, vb = (item.clone() for item in (q, k, v))
            start, end = ref_range(info, 1)
            KleinRefLatentControllerKVCallback(0.25, 1, "none", 0.5)(q, k, v, info)
            self.assertTrue(torch.equal(q, qb))
            self.assertTrue(torch.equal(k[:, :, start:end], kb[:, :, start:end] * 0.25))
            self.assertTrue(torch.equal(v[:, :, start:end], vb[:, :, start:end] * 0.25))
            mask = torch.ones(k.shape[2], dtype=torch.bool); mask[start:end] = False
            self.assertTrue(torch.equal(k[:, :, mask], kb[:, :, mask]))
            self.assertTrue(torch.equal(v[:, :, mask], vb[:, :, mask]))

    def test_neutral_no_write_and_zero_reference_noop(self):
        for info, mode, fade in (
            (metadata("double"), "none", 0.5),
            (metadata("double"), "top_down", 0.0),
            (metadata("single", (), ()), "none", 0.5),
        ):
            q, k, v = tensors(info); before = tuple(item.clone() for item in (q, k, v))
            KleinRefLatentControllerKVCallback(1.0, 0, mode, fade)(q, k, v, info)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip((q, k, v), before, strict=True)))

    def test_original_all_block_schedule_has_no_depth_boundaries(self):
        for block_type, indexes in (("double", (0, 7)), ("single", (0, 23))):
            for block_index in indexes:
                info = metadata(block_type); info.block_index = block_index
                q, k, v = tensors(info); kb = k.clone(); start, end = ref_range(info, 0)
                KleinRefLatentControllerKVCallback(2.0, 0, "none", 0.5)(q, k, v, info)
                self.assertTrue(torch.equal(k[:, :, start:end], kb[:, :, start:end] * 2.0))

    def test_spatial_modes_orientation_and_non_square_flattening(self):
        info = metadata("single", refs=(6,), shapes=((2, 3),))
        expected = {
            "top_down": torch.tensor([1, 1, 1, 0, 0, 0.0]),
            "left_right": torch.tensor([1, .5, 0, 1, .5, 0.0]),
        }
        for mode, weights in expected.items():
            q, k, v = tensors(info); kb, vb = k.clone(), v.clone()
            start, end = ref_range(info, 0)
            KleinRefLatentControllerKVCallback(1.0, 0, mode, 1.0)(q, k, v, info)
            scale = weights.view(1, 1, -1, 1)
            self.assertTrue(torch.equal(k[:, :, start:end], kb[:, :, start:end] * scale))
            self.assertTrue(torch.equal(v[:, :, start:end], vb[:, :, start:end] * scale))

    def test_radial_fade_endpoints(self):
        info = metadata("double", refs=(9,), shapes=((3, 3),))
        for mode, center, corner in (("center_out", 1.0, 0.0), ("edges_out", 0.0, 1.0)):
            q, k, v = tensors(info); kb = k.clone(); start, end = ref_range(info, 0)
            KleinRefLatentControllerKVCallback(1.0, 0, mode, 1.0)(q, k, v, info)
            ratios = k[0, 0, start:end, 0] / kb[0, 0, start:end, 0]
            self.assertAlmostEqual(float(ratios[4]), center, places=6)
            self.assertAlmostEqual(float(ratios[0]), corner, places=6)

    def test_invalid_index_or_shape_fails_before_mutation(self):
        cases = [(metadata("single"), KleinRefLatentControllerKVCallback(0.0, 2, "none", 0.5), IndexError)]
        bad = metadata("single", refs=(6,), shapes=((1, 5),))
        cases.append((bad, KleinRefLatentControllerKVCallback(0.0, 0, "none", 0.5), ValueError))
        for info, callback, error in cases:
            q, k, v = tensors(info); before = tuple(item.clone() for item in (q, k, v))
            with self.assertRaises(error): callback(q, k, v, info)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip((q, k, v), before, strict=True)))

    def test_multiplicative_stacking_with_existing_direct_callbacks(self):
        info = metadata("double", refs=(6,), shapes=((2, 3),))
        q, k, v = tensors(info); kb = k.clone(); start, end = ref_range(info, 0)
        KleinTextRefBalanceKVCallback(1.0, 0.8)(q, k, v, info)
        KleinRefLatentWeightKVCallback(0, 0.5)(q, k, v, info)
        KleinRefLatentControllerKVCallback(1.25, 0, "none", 0.5)(q, k, v, info)
        self.assertTrue(torch.allclose(k[:, :, start:end], kb[:, :, start:end] * 0.5))

    def test_prediction_space_mixing_is_rejected_in_both_graph_orders(self):
        source = FakeModelPatcher(make_adapter())
        prediction = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeight"]().execute(source, 0, 0.5)[0]
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.node.control(prediction, self.conditioning)

        direct = self.node.control(source, self.conditioning)[0]
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeight"]().execute(direct, 0, 0.5)


if __name__ == "__main__":
    unittest.main()
