import importlib.util
import pathlib
import sys
import unittest

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_mask_reference_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_mask_reference_test.nodes.enhancer.mask_reference import (
    NunchakuKleinMaskReferenceController,
)


def conditioning(references):
    metadata = {
        "marker": object(),
        "reference_latents": list(references),
        "reference_latents_method": "index",
    }
    return [[torch.zeros((1, 4, 8)), metadata]]


class MaskReferenceControllerTests(unittest.TestCase):
    def setUp(self):
        self.node = NunchakuKleinMaskReferenceController()

    def test_ui_contract_and_optional_inputs(self):
        inputs = self.node.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["conditioning", "mask"])
        self.assertEqual(
            list(inputs["optional"]),
            ["strength", "invert_mask", "feather", "reference_index", "debug"],
        )

    def test_one_reference_zero_and_one_masks_without_source_mutation(self):
        reference = torch.arange(24, dtype=torch.float16).reshape(1, 2, 3, 4)
        source = conditioning([reference])
        source_snapshot = reference.clone()
        zero_mask = torch.zeros((3, 4))
        one_mask = torch.ones((3, 4))

        zero_output = self.node.apply_mask(source, zero_mask, strength=1.0)[0]
        one_output = self.node.apply_mask(source, one_mask, strength=1.0)[0]
        zero_reference = zero_output[0][1]["reference_latents"][0]
        one_reference = one_output[0][1]["reference_latents"][0]

        self.assertTrue(torch.equal(zero_reference, torch.zeros_like(reference)))
        self.assertTrue(torch.equal(one_reference, reference))
        self.assertEqual(zero_reference.dtype, reference.dtype)
        self.assertNotEqual(
            zero_reference.untyped_storage().data_ptr(),
            reference.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            one_reference.untyped_storage().data_ptr(),
            reference.untyped_storage().data_ptr(),
        )
        self.assertTrue(torch.equal(reference, source_snapshot))
        self.assertTrue(torch.equal(zero_mask, torch.zeros_like(zero_mask)))
        self.assertTrue(torch.equal(one_mask, torch.ones_like(one_mask)))
        self.assertIs(source[0][1]["reference_latents"][0], reference)
        self.assertIsNot(zero_output[0][1], source[0][1])
        self.assertEqual(zero_output[0][1]["reference_latents_method"], "index")

    def test_multiple_mixed_references_preserve_order_and_replace_only_selected(self):
        references = (
            torch.full((1, 2, 2, 3), 1.0),
            torch.full((1, 2, 4, 1), 2.0),
            torch.full((1, 2, 1, 5), 3.0),
        )
        source = conditioning(references)
        output = self.node.apply_mask(
            source, torch.zeros((2, 2)), strength=0.5, reference_index=1
        )[0]
        updated = output[0][1]["reference_latents"]

        self.assertIs(updated[0], references[0])
        self.assertIs(updated[2], references[2])
        self.assertTrue(torch.equal(updated[1], torch.full_like(references[1], 1.0)))
        self.assertNotEqual(
            updated[1].untyped_storage().data_ptr(),
            references[1].untyped_storage().data_ptr(),
        )
        self.assertEqual([list(value.shape) for value in updated], [
            [1, 2, 2, 3], [1, 2, 4, 1], [1, 2, 1, 5]
        ])

    def test_reference_batch_broadcasts_first_mask_plane(self):
        reference = torch.stack(
            (torch.ones((2, 2, 3)), torch.full((2, 2, 3), 2.0))
        )
        mask = torch.stack((torch.zeros((2, 3)), torch.ones((2, 3))))
        output = self.node.apply_mask(conditioning([reference]), mask)[0]
        updated = output[0][1]["reference_latents"][0]
        self.assertEqual(list(updated.shape), [2, 2, 2, 3])
        self.assertTrue(torch.equal(updated, torch.zeros_like(reference)))

    def test_resize_invert_feather_and_neutral_strength(self):
        reference = torch.ones((1, 1, 4, 6))
        source = conditioning([reference])
        mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        output = self.node.apply_mask(
            source, mask, strength=0.75, invert_mask=True, feather=1
        )[0]
        updated = output[0][1]["reference_latents"][0]
        self.assertEqual(list(updated.shape), [1, 1, 4, 6])
        self.assertTrue(torch.isfinite(updated).all())
        self.assertGreaterEqual(float(updated.min()), 0.25)
        self.assertLessEqual(float(updated.max()), 1.0)
        self.assertIs(self.node.apply_mask(source, mask, strength=0.0)[0], source)

    def test_missing_selected_reference_returns_copied_unchanged_conditioning(self):
        reference = torch.ones((1, 2, 2, 2))
        source = conditioning([reference])
        output = self.node.apply_mask(
            source, torch.ones((2, 2)), reference_index=3
        )[0]
        self.assertIsNot(output, source)
        self.assertIsNot(output[0][1], source[0][1])
        self.assertIs(output[0][1]["reference_latents"][0], reference)

    def test_malformed_inputs_fail_closed(self):
        reference = torch.ones((1, 2, 2, 2))
        source = conditioning([reference])
        cases = (
            (None, torch.ones((2, 2)), {}, TypeError, "conditioning"),
            ([[object(), {}]], torch.ones((2, 2)), {}, TypeError, "tensor"),
            (conditioning([object()]), torch.ones((2, 2)), {}, TypeError, "torch.Tensor"),
            (conditioning([torch.ones((2, 2, 2))]), torch.ones((2, 2)), {}, ValueError, "BCHW"),
            (source, object(), {}, TypeError, "mask must be a torch.Tensor"),
            (source, torch.ones((1,)), {}, ValueError, "rank 2, 3, or 4"),
            (source, torch.empty((0, 2)), {}, ValueError, "non-empty"),
            (source, torch.full((2, 2), 2.0), {}, ValueError, r"within \[0, 1\]"),
            (source, torch.ones((2, 2)), {"strength": 1.1}, ValueError, "strength"),
            (source, torch.ones((2, 2)), {"feather": 65}, ValueError, "feather"),
            (source, torch.ones((2, 2)), {"reference_index": 8}, ValueError, "reference_index"),
        )
        for conditioning_value, mask, kwargs, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    self.node.apply_mask(conditioning_value, mask, **kwargs)


if __name__ == "__main__":
    unittest.main()
