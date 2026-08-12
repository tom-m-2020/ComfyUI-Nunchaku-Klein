import copy
from dataclasses import FrozenInstanceError
import importlib.util
import pathlib
from types import SimpleNamespace
import sys
import tempfile
import unittest
import weakref
import gc

import torch
from torch import nn


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_identity_final_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_identity_final_test.models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    IDENTITY_FEATURE_TRANSFER_FINAL_OPTION,
    NunchakuFlux2KleinAdapter,
)
from nunchaku_klein_identity_final_test.nodes.enhancer.common import REFERENCE_CATEGORY
from nunchaku_klein_identity_final_test.nodes.enhancer.identity_feature_transfer_final import (
    HARD_DOUBLE,
    HARD_SINGLE,
    KleinIdentityFeatureTransferFinalCallback,
    _evenly_spaced_indices,
    _parse_schedule,
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


def metadata(block_type="double", block_index=0, refs=(2, 3), shapes=((1, 2), (1, 3))):
    text, generated = 2, 3
    logical_image = generated + sum(refs)
    if block_type == "double":
        padded_text, padded_image = 4, max(8, logical_image)
    else:
        padded_text, padded_image = text, max(10, logical_image)
    return SimpleNamespace(
        block_type=block_type, block_index=block_index,
        text_token_count=text, generated_token_count=generated,
        generated_spatial_shape=(1, generated),
        reference_token_counts=refs, reference_spatial_shapes=shapes,
        logical_image_token_count=logical_image,
        padded_text_token_count=padded_text, padded_image_token_count=padded_image,
        packed_sequence_length=padded_text + padded_image,
        batch_size=1, head_count=2, head_dimension=2,
    )


def ranges(info):
    image_start = info.padded_text_token_count if info.block_type == "double" else info.text_token_count
    generated = (image_start, image_start + info.generated_token_count)
    refs = []
    position = generated[1]
    for count in info.reference_token_counts:
        refs.append((position, position + count)); position += count
    return generated, tuple(refs)


def callback(*, selected=None, double=(1.0,) + (0.0,) * 7, single=(1.0,) + (0.0,) * 23,
             floor=0.0, temperature=0.1, threshold=1.0, masks=(None,) * 8):
    return KleinIdentityFeatureTransferFinalCallback(
        selected, double, single, floor, temperature, threshold, masks, False
    )


def independent_final(output, info, selected, strength, floor, temperature, bank_indices=None, reference_override=None):
    generated_range, ref_ranges = ranges(info)
    gen = output[:, generated_range[0]:generated_range[1]]
    ref = torch.cat([output[:, ref_ranges[index][0]:ref_ranges[index][1]] for index in selected], 1)
    if reference_override is not None:
        ref = reference_override
    if bank_indices is not None:
        ref = ref.index_select(1, torch.tensor(bank_indices))
    gf, rf = gen.float(), ref.float()
    gn = torch.nn.functional.normalize(gf - gf.mean(1, keepdim=True), dim=-1)
    rn = torch.nn.functional.normalize(rf - rf.mean(1, keepdim=True), dim=-1)
    sim = torch.bmm(gn, rn.transpose(1, 2))
    sim = torch.where(sim >= floor, sim, torch.full_like(sim, torch.finfo(sim.dtype).min))
    weights = torch.nan_to_num(torch.softmax(sim / temperature, -1), nan=0.0)
    pooled = torch.bmm(weights, rf)
    best = sim.max(-1).values
    best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
    confidence = ((best - floor) / max(1 - floor, 1e-6)).clamp(0, 1)
    result = output.clone()
    result[:, generated_range[0]:generated_range[1]] = gen + (
        pooled.to(gen.dtype) - gen
    ) * (confidence * strength).unsqueeze(-1).to(gen.dtype)
    return result


class IdentityFeatureTransferFinalTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinIdentityFeatureTransferFinal"]()

    def test_registration_ui_clone_callbacks_and_frozen_config(self):
        cls = type(self.node); inputs = cls.INPUT_TYPES()
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL",)); self.assertEqual(cls.FUNCTION, "apply")
        self.assertEqual(list(inputs["required"]), [
            "model", "preset", "enabled", "reference_index", "reference_indices",
            "similarity_floor", "softmax_temperature", "mask_threshold",
            "double_blocks", "single_blocks", "debug", "mask_behavior",
        ])
        self.assertEqual(list(inputs["optional"]), [
            "sigmas", "debug_spatial", "debug_probe_block_type",
            "debug_probe_block_index", "debug_eligible_bank_cap",
            "debug_reference_pool_height", "debug_reference_pool_width",
            *[f"subject_mask_{i}" for i in range(1, 9)],
        ])
        self.assertEqual(inputs["required"]["double_blocks"][1]["default"], HARD_DOUBLE)
        self.assertEqual(inputs["required"]["single_blocks"][1]["default"], HARD_SINGLE)
        source = FakeModelPatcher(make_adapter())
        pre, post = lambda *args: None, lambda *args: None
        source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION] = Flux2AttentionCallbacks((pre,), (post,))
        branch = self.node.apply(source, preset="custom", double_blocks="0:1", single_blocks="")[0]
        self.assertIsNot(branch, source)
        source_callbacks = source.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION]
        branch_callbacks = branch.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION]
        self.assertEqual(source_callbacks, Flux2AttentionCallbacks((pre,), (post,)))
        self.assertEqual(branch_callbacks.pre_attention_callbacks, (pre,))
        self.assertEqual(branch_callbacks.post_attention_callbacks[0], post)
        final = branch_callbacks.post_attention_callbacks[-1]
        self.assertIs(branch.model_options["transformer_options"][IDENTITY_FEATURE_TRANSFER_FINAL_OPTION][0], final)
        with self.assertRaises(FrozenInstanceError): final.similarity_floor = 0.2

    def test_exact_algorithm_and_non_generated_regions_for_both_streams(self):
        for stream in ("double", "single"):
            info = metadata(stream)
            output = torch.tensor(range(info.packed_sequence_length * 4), dtype=torch.float32).reshape(1, info.packed_sequence_length, 4) / 10
            before = output.clone()
            actual = callback(selected=(0, 1))(output, info)
            expected = independent_final(before, info, (0, 1), 1.0, 0.0, 0.1)
            self.assertTrue(torch.equal(actual, expected))
            gen, refs = ranges(info); outside = torch.ones(info.packed_sequence_length, dtype=torch.bool)
            outside[gen[0]:gen[1]] = False
            self.assertTrue(torch.equal(actual[:, outside], before[:, outside]))
            self.assertTrue(torch.equal(output, before))

    def test_inactive_blocks_are_exact_noops_and_schedules_are_independent_inclusive(self):
        self.assertEqual(_parse_schedule("2-4:mid_img=0.5", 8, name="double"), (0, 0, .5, .5, .5, 0, 0, 0))
        cb = callback(double=(0.0,) * 8, single=(0.0, 0.0, 1.0) + (0.0,) * 21)
        for info in (metadata("double", 2), metadata("single", 1)):
            output = torch.randn(1, info.packed_sequence_length, 4); before = output.clone()
            self.assertIsNone(cb(output, info)); self.assertTrue(torch.equal(output, before))
        info = metadata("single", 2); output = torch.randn(1, info.packed_sequence_length, 4)
        self.assertIsNotNone(cb(output, info))

    def test_global_bank_selection_order_and_invalid_index(self):
        cb = callback(selected=(1, 0))
        self.assertEqual(cb.selected_reference_indices, (1, 0))
        info = metadata(); output = torch.randn(1, info.packed_sequence_length, 4)
        self.assertIsNotNone(cb(output, info))
        before = output.clone()
        with self.assertRaises(IndexError): callback(selected=(2,))(output, info)
        self.assertTrue(torch.equal(output, before))

    def test_focus_masks_authoritative_non_square_shapes_and_empty_bank(self):
        info = metadata(refs=(6, 4), shapes=((2, 3), (1, 4)))
        output = torch.randn(1, info.packed_sequence_length, 4)
        first_only = torch.tensor([[1.0, 0, 0], [0, 0, 0]])
        cb = callback(selected=(0,), masks=(first_only,) + (None,) * 7)
        self.assertIsNotNone(cb(output, info))
        empty = callback(selected=(0,), masks=(torch.zeros(2, 3),) + (None,) * 7)
        before = output.clone(); self.assertIsNone(empty(output, info)); self.assertTrue(torch.equal(output, before))
        bad = metadata(refs=(6,), shapes=((1, 5),)); bad_output = torch.randn(1, bad.packed_sequence_length, 4)
        with self.assertRaises(ValueError): callback(selected=(0,))(bad_output, bad)

    def test_dtype_restoration_batch_and_no_feature_retention(self):
        info = metadata("single")
        info.batch_size = 2
        for dtype in (torch.float16, torch.bfloat16):
            output = torch.randn(2, info.packed_sequence_length, 4, dtype=dtype)
            result = callback(selected=(0,))(output, info)
            self.assertEqual(result.dtype, dtype); self.assertEqual(result.shape, output.shape)
        ref = weakref.ref(result); del result; gc.collect()
        self.assertIsNone(ref())

    def test_disabled_zero_unmasked_sigma_and_validation(self):
        source = FakeModelPatcher(make_adapter())
        branch = self.node.apply(source, enabled=False)[0]
        self.assertIsNot(branch, source)
        self.assertNotIn(ATTENTION_CALLBACKS_OPTION, branch.model_options["transformer_options"])
        with self.assertRaisesRegex(NotImplementedError, "source exclusion"):
            self.node.apply(source, mask_behavior="zero_unmasked_tokens")
        with self.assertRaisesRegex(NotImplementedError, "sampling sigma"):
            self.node.apply(source, sigmas=torch.tensor([1.0, 0.0]))
        with self.assertRaises(ValueError): self.node.apply(source, preset="custom", softmax_temperature=0.0)
        with self.assertRaises(ValueError): self.node.apply(source, reference_indices="bogus")

    def test_callback_has_no_cross_call_feature_state(self):
        info = metadata(); cb = callback(selected=(0,))
        first = torch.randn(1, info.packed_sequence_length, 4)
        second = torch.randn(1, info.packed_sequence_length, 4)
        first_result = cb(first, info); second_result = cb(second, info)
        self.assertTrue(torch.equal(first_result, independent_final(first, info, (0,), 1.0, 0.0, 0.1)))
        self.assertTrue(torch.equal(second_result, independent_final(second, info, (0,), 1.0, 0.0, 0.1)))

    def test_debug_summaries_report_matching_transfer_and_empty_mask(self):
        info = metadata()
        output = torch.randn(1, info.packed_sequence_length, 4)
        active = KleinIdentityFeatureTransferFinalCallback(
            (0,), (1.0,) + (0.0,) * 7, (0.0,) * 24,
            0.0, 0.1, 0.5, (torch.tensor([[1.0, 0.0]]),) + (None,) * 7, True,
        )
        with self.assertLogs(active.__class__.__module__, level="INFO") as captured:
            self.assertIsNotNone(active(output, info))
        messages = "\n".join(captured.output)
        for expected in (
            "IFT Final mask: double 0 ref=0",
            "eligible=1/2",
            "IFT Final: double 0 refs=(0,)",
            "bank=1 generated=3",
            "similarity: mean=",
            "above_floor=",
            "confidence: mean=",
            "transfer: mean_delta_norm=",
            "relative_delta=",
        ):
            self.assertIn(expected, messages)

        empty = KleinIdentityFeatureTransferFinalCallback(
            (0,), (1.0,) + (0.0,) * 7, (0.0,) * 24,
            0.0, 0.1, 1.0, (torch.zeros(1, 2),) + (None,) * 7, True,
        )
        with self.assertLogs(empty.__class__.__module__, level="INFO") as captured:
            self.assertIsNone(empty(output, info))
        messages = "\n".join(captured.output)
        self.assertIn("eligible=0/2", messages)
        self.assertIn("eligible bank is empty -> no-op", messages)

    def test_spatial_debug_probe_writes_only_the_selected_block(self):
        info = metadata()
        output = torch.randn(1, info.packed_sequence_length, 4)
        with tempfile.TemporaryDirectory() as directory:
            probe = KleinIdentityFeatureTransferFinalCallback(
                (0,), (1.0,) + (0.0,) * 7, (0.0,) * 24,
                0.0, 0.1, 1.0, (None,) * 8, True,
                True, "double", 0, directory,
            )
            with self.assertLogs(probe.__class__.__module__, level="INFO") as captured:
                probe(output, info)
            files = tuple(pathlib.Path(directory).glob("*.png"))
            self.assertEqual(len(files), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in files))
            messages = "\n".join(captured.output)
            for expected in (
                "reference_tokens_total=2 eligible_reference_tokens=2 matching_reference_tokens=2",
                "best_similarity: mean=",
                "p90=",
                "p95=",
                "p99=",
                "confidence: mean=",
                "delta_norm: mean=",
                "top=1% bbox_area_fraction=",
            ):
                self.assertIn(expected, messages)

            other_block = metadata(block_index=1)
            probe(output, other_block)
            self.assertEqual(len(tuple(pathlib.Path(directory).glob("*.png"))), 4)

    def test_diagnostic_bank_cap_is_uniform_deterministic_and_feature_aligned(self):
        indices = _evenly_spaced_indices(10, 4, torch.device("cpu"))
        self.assertEqual(indices.tolist(), [0, 3, 6, 9])
        self.assertEqual(
            _evenly_spaced_indices(1080, 284, torch.device("cpu"))[[0, -1]].tolist(),
            [0, 1079],
        )
        self.assertEqual(
            torch.unique(_evenly_spaced_indices(1080, 284, torch.device("cpu"))).numel(),
            284,
        )

        info = metadata(refs=(10,), shapes=((2, 5),))
        output = torch.randn(1, info.packed_sequence_length, 4)
        capped = KleinIdentityFeatureTransferFinalCallback(
            (0,), (1.0,) + (0.0,) * 7, (0.0,) * 24,
            0.0, 0.1, 1.0, (None,) * 8, True,
            False, "double", 0, None, 4,
        )
        with self.assertLogs(capped.__class__.__module__, level="INFO") as captured:
            actual = capped(output, info)
        expected = independent_final(output, info, (0,), 1.0, 0.0, 0.1, [0, 3, 6, 9])
        self.assertTrue(torch.equal(actual, expected))
        messages = "\n".join(captured.output)
        self.assertIn("eligible=10 capped=4 first=0 last=9", messages)

    def test_diagnostic_bank_cap_requires_debug(self):
        source = FakeModelPatcher(make_adapter())
        with self.assertRaisesRegex(ValueError, "requires debug=true"):
            self.node.apply(source, debug=False, debug_eligible_bank_cap=284)

    def test_diagnostic_feature_pool_uses_spatial_grid_and_pooled_mask(self):
        info = metadata(refs=(16,), shapes=((4, 4),))
        output = torch.randn(1, info.packed_sequence_length, 4)
        _, ref_ranges = ranges(info)
        start, end = ref_ranges[0]
        source_reference = output[:, start:end]
        pooled_reference = torch.nn.functional.adaptive_avg_pool2d(
            source_reference.float().reshape(1, 4, 4, 4).permute(0, 3, 1, 2),
            (2, 2),
        ).permute(0, 2, 3, 1).reshape(1, 4, 4)
        mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        callback = KleinIdentityFeatureTransferFinalCallback(
            (0,), (1.0,) + (0.0,) * 7, (0.0,) * 24,
            0.0, 0.1, 0.5, (mask,) + (None,) * 7, True,
            False, "double", 0, None, 0, (2, 2),
        )
        with self.assertLogs(callback.__class__.__module__, level="INFO") as captured:
            actual = callback(output, info)
        expected = independent_final(
            output, info, (0,), 1.0, 0.0, 0.1,
            reference_override=pooled_reference[:, :1],
        )
        self.assertTrue(torch.equal(actual, expected))
        messages = "\n".join(captured.output)
        self.assertIn("source=(4,4) target=(2,2) tokens=16->4", messages)
        self.assertIn("eligible=1/4", messages)

    def test_diagnostic_feature_pool_validation(self):
        source = FakeModelPatcher(make_adapter())
        with self.assertRaisesRegex(ValueError, "both be zero or both be positive"):
            self.node.apply(source, debug=True, debug_reference_pool_height=32)
        with self.assertRaisesRegex(ValueError, "requires debug=true"):
            self.node.apply(
                source, debug=False,
                debug_reference_pool_height=32, debug_reference_pool_width=32,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.node.apply(
                source, debug=True, debug_eligible_bank_cap=284,
                debug_reference_pool_height=32, debug_reference_pool_width=32,
            )


if __name__ == "__main__":
    unittest.main()
