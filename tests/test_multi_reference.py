import importlib.util
import pathlib
import sys
import unittest

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_multi_reference_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_multi_reference_test.nodes.enhancer.multi_reference import (
    NunchakuKleinMultiReferenceLatent,
)


def conditioning():
    old_reference = torch.full((1, 128, 1, 1), -1.0)
    metadata = {
        "marker": "original",
        "reference_latents": [old_reference],
        "reference_latents_method": "old",
    }
    return [[torch.zeros((1, 4, 8)), metadata]], old_reference


class MultiReferenceLatentTests(unittest.TestCase):
    def setUp(self):
        self.node = NunchakuKleinMultiReferenceLatent()

    def test_ui_contract_is_one_required_and_seven_optional_latents(self):
        inputs = self.node.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["conditioning", "latent_1"])
        self.assertEqual(
            list(inputs["optional"]),
            [f"latent_{index}" for index in range(2, 9)],
        )
        self.assertTrue(all(value == ("LATENT",) for value in inputs["optional"].values()))

    def test_one_reference_replaces_metadata_without_mutating_input(self):
        source, old_reference = conditioning()
        reference = torch.ones((1, 128, 2, 3))
        output = self.node.apply(source, {"samples": reference})[0]

        self.assertIsNot(output, source)
        self.assertIsNot(output[0], source[0])
        self.assertIsNot(output[0][1], source[0][1])
        self.assertEqual(output[0][1]["marker"], "original")
        self.assertEqual(output[0][1]["reference_latents_method"], "index")
        self.assertEqual(len(output[0][1]["reference_latents"]), 1)
        output_reference = output[0][1]["reference_latents"][0]
        self.assertTrue(torch.equal(output_reference, reference))
        self.assertEqual(
            output_reference.untyped_storage().data_ptr(),
            reference.untyped_storage().data_ptr(),
        )
        self.assertFalse(output_reference.requires_grad)
        self.assertIs(source[0][1]["reference_latents"][0], old_reference)
        self.assertEqual(source[0][1]["reference_latents_method"], "old")

    def test_batched_and_mixed_spatial_inputs_split_in_exact_order(self):
        source, _ = conditioning()
        first = torch.stack(
            (torch.full((128, 2, 3), 1.0), torch.full((128, 2, 3), 2.0))
        )
        second = torch.full((1, 128, 4, 1), 3.0)
        third = torch.stack(
            (torch.full((128, 1, 5), 4.0), torch.full((128, 1, 5), 5.0))
        )
        output = self.node.apply(
            source,
            {"samples": first},
            latent_2={"samples": second},
            latent_4={"samples": third},
        )[0]
        references = output[0][1]["reference_latents"]

        self.assertEqual([list(value.shape) for value in references], [
            [1, 128, 2, 3], [1, 128, 2, 3], [1, 128, 4, 1],
            [1, 128, 1, 5], [1, 128, 1, 5],
        ])
        self.assertEqual([float(value[0, 0, 0, 0]) for value in references], [1, 2, 3, 4, 5])
        self.assertTrue(all(value.shape[0] == 1 and not value.requires_grad for value in references))
        self.assertEqual(output[0][1]["reference_latents_method"], "index")

    def test_malformed_conditioning_and_latents_fail_closed(self):
        source, _ = conditioning()
        cases = (
            (None, {"samples": torch.ones((1, 128, 1, 1))}, TypeError, "conditioning"),
            ([[object(), {}]], {"samples": torch.ones((1, 128, 1, 1))}, TypeError, "tensor"),
            (source, {}, ValueError, "samples"),
            (source, {"samples": object()}, TypeError, "torch.Tensor"),
            (source, {"samples": torch.ones((128, 2, 2))}, ValueError, "BCHW"),
            (source, {"samples": torch.ones((1, 128, 0, 2))}, ValueError, "positive"),
        )
        for conditioning_value, latent, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    self.node.apply(conditioning_value, latent)


if __name__ == "__main__":
    unittest.main()
