import importlib.util
import sys
import unittest
from pathlib import Path

import torch


TARGET = Path(__file__).resolve().parents[1]
COMFY = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")


def load_target_package():
    sys.path.insert(0, str(COMFY))
    spec = importlib.util.spec_from_file_location(
        "nunchaku_klein_enhancer_test",
        TARGET / "__init__.py",
        submodule_search_locations=[str(TARGET)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


PACKAGE = load_target_package()
ENHANCER = sys.modules[f"{PACKAGE.__name__}.nodes.enhancer"]
COMMON = sys.modules[f"{PACKAGE.__name__}.nodes.enhancer.common"]


class KleinEnhancerTests(unittest.TestCase):
    def test_klein_enhancer_registration_category_and_ui_contract(self):
        node_id = "NunchakuKleinEnhancer"
        self.assertIn(node_id, PACKAGE.NODE_CLASS_MAPPINGS)
        node = PACKAGE.NODE_CLASS_MAPPINGS[node_id]
        self.assertEqual(node.CATEGORY, COMMON.TEXT_CATEGORY)
        self.assertEqual(node.RETURN_TYPES, ("CONDITIONING",))
        self.assertEqual(node.FUNCTION, "enhance")

        inputs = node.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["conditioning", "active_scale", "per_token_whiten", "norm_equalize"],
        )
        self.assertEqual(
            list(inputs["optional"]),
            [
                "early_layer_scale",
                "mid_layer_scale",
                "late_layer_scale",
                "preserve_original",
                "active_end_override",
                "device",
                "debug",
            ],
        )
        expected_numeric = {
            "active_scale": (1.0, 0.0, 10.0, 0.05),
            "per_token_whiten": (0.0, -1.0, 5.0, 0.05),
            "norm_equalize": (0.0, 0.0, 1.0, 0.05),
            "early_layer_scale": (1.0, 0.0, 5.0, 0.05),
            "mid_layer_scale": (1.0, 0.0, 5.0, 0.05),
            "late_layer_scale": (1.0, 0.0, 5.0, 0.05),
            "preserve_original": (0.0, 0.0, 1.0, 0.05),
            "active_end_override": (0, 0, 512, 1),
        }
        for name, expected in expected_numeric.items():
            group = "required" if name in inputs["required"] else "optional"
            options = inputs[group][name][1]
            self.assertEqual(
                (options["default"], options["min"], options["max"], options["step"]),
                expected,
            )
            self.assertIn("tooltip", options)
        self.assertEqual(inputs["optional"]["device"][0][:2], ["auto", "cpu"])
        self.assertEqual(inputs["optional"]["device"][1], {"default": "auto"})
        self.assertEqual(inputs["optional"]["debug"], ("BOOLEAN", {"default": False}))

    def test_klein_enhancer_neutral_is_exact_passthrough(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.randn((2, 4, 12288), dtype=torch.float16)
        metadata = {"attention_mask": torch.ones((2, 4))}
        conditioning = [[tensor, metadata]]

        output = node.enhance(
            conditioning,
            active_scale=1.0,
            per_token_whiten=0.0,
            norm_equalize=0.0,
        )[0]

        self.assertIs(output, conditioning)
        self.assertIs(output[0][0], tensor)
        self.assertIs(output[0][1], metadata)

    def test_klein_enhancer_active_scale_preserves_input_dtype_batch_and_inactive(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.arange(2 * 4 * 12288, dtype=torch.float32).reshape(
            2, 4, 12288
        ).to(torch.float16)
        snapshot = tensor.clone()
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        metadata = {"attention_mask": attention_mask, "marker": "original"}
        conditioning = [[tensor, metadata]]

        output = node.enhance(
            conditioning,
            active_scale=2.0,
            device="cpu",
        )[0]
        result = output[0][0]

        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape, tensor.shape)
        self.assertTrue(torch.equal(result[:, :2], snapshot[:, :2] * 2.0))
        self.assertTrue(torch.equal(result[:, 2:], snapshot[:, 2:]))
        self.assertTrue(torch.equal(tensor, snapshot))
        self.assertIs(conditioning[0][1], metadata)
        self.assertIsNot(output[0][1], metadata)
        self.assertEqual(output[0][1]["marker"], "original")
        self.assertIs(output[0][1]["attention_mask"], attention_mask)

    def test_klein_enhancer_whitening_scales_deviation_from_sequence_mean(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.empty((1, 3, 12288), dtype=torch.float32)
        tensor[:, 0].fill_(1.0)
        tensor[:, 1].fill_(3.0)
        tensor[:, 2].fill_(5.0)

        result = node.enhance(
            [[tensor, {}]],
            per_token_whiten=1.0,
            device="cpu",
        )[0][0][0]

        self.assertTrue(torch.equal(result[:, 0], torch.full_like(result[:, 0], -1.0)))
        self.assertTrue(torch.equal(result[:, 1], torch.full_like(result[:, 1], 3.0)))
        self.assertTrue(torch.equal(result[:, 2], torch.full_like(result[:, 2], 7.0)))

    def test_klein_enhancer_norm_equalize_targets_global_active_mean_norm(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.empty((1, 2, 12288), dtype=torch.float32)
        tensor[:, 0].fill_(1.0)
        tensor[:, 1].fill_(3.0)

        result = node.enhance(
            [[tensor, {}]],
            norm_equalize=1.0,
            device="cpu",
        )[0][0][0]
        norms = result.norm(dim=-1)
        expected = tensor.norm(dim=-1).mean()

        self.assertTrue(torch.allclose(norms, torch.full_like(norms, expected)))

    def test_klein_enhancer_12288_layer_slice_boundaries_are_exact(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.ones((1, 1, 12288), dtype=torch.float32)

        result = node.enhance(
            [[tensor, {}]],
            early_layer_scale=2.0,
            mid_layer_scale=3.0,
            late_layer_scale=4.0,
            device="cpu",
        )[0][0][0]

        self.assertTrue(torch.equal(result[:, :, :4096], torch.full_like(result[:, :, :4096], 2.0)))
        self.assertTrue(torch.equal(result[:, :, 4096:8192], torch.full_like(result[:, :, 4096:8192], 3.0)))
        self.assertTrue(torch.equal(result[:, :, 8192:], torch.full_like(result[:, :, 8192:], 4.0)))

    def test_klein_enhancer_7680_layer_slice_boundaries_match_source_contract(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.ones((1, 1, 7680), dtype=torch.float32)

        result = node.enhance(
            [[tensor, {}]],
            early_layer_scale=2.0,
            mid_layer_scale=3.0,
            late_layer_scale=4.0,
            device="cpu",
        )[0][0][0]

        self.assertTrue(torch.equal(result[:, :, :2560], torch.full_like(result[:, :, :2560], 2.0)))
        self.assertTrue(torch.equal(result[:, :, 2560:5120], torch.full_like(result[:, :, 2560:5120], 3.0)))
        self.assertTrue(torch.equal(result[:, :, 5120:], torch.full_like(result[:, :, 5120:], 4.0)))

    def test_klein_enhancer_preserve_original_endpoints_and_intermediate(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.ones((1, 2, 12288), dtype=torch.float32)

        full = node.enhance(
            [[tensor, {}]], active_scale=3.0, preserve_original=0.0, device="cpu"
        )[0][0][0]
        half = node.enhance(
            [[tensor, {}]], active_scale=3.0, preserve_original=0.5, device="cpu"
        )[0][0][0]
        original = node.enhance(
            [[tensor, {}]], active_scale=3.0, preserve_original=1.0, device="cpu"
        )[0][0][0]

        self.assertTrue(torch.equal(full, torch.full_like(full, 3.0)))
        self.assertTrue(torch.equal(half, torch.full_like(half, 2.0)))
        self.assertTrue(torch.equal(original, tensor))

    def test_klein_enhancer_override_precedes_attention_mask(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.ones((1, 5, 12288), dtype=torch.float32)
        metadata = {"attention_mask": torch.tensor([[1, 0, 0, 0, 0]])}

        result = node.enhance(
            [[tensor, metadata]],
            active_scale=2.0,
            active_end_override=3,
            device="cpu",
        )[0][0][0]

        self.assertTrue(torch.equal(result[:, :3], tensor[:, :3] * 2.0))
        self.assertTrue(torch.equal(result[:, 3:], tensor[:, 3:]))

    def test_klein_enhancer_attention_mask_and_fallback_match_source(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.ones((1, 5, 12288), dtype=torch.float32)

        masked = node.enhance(
            [[tensor, {"attention_mask": torch.tensor([[1, 1, 0, 0, 0]])}]],
            active_scale=2.0,
            device="cpu",
        )[0][0][0]
        missing = node.enhance(
            [[tensor, {}]], active_scale=2.0, device="cpu"
        )[0][0][0]
        invalid_rank = node.enhance(
            [[tensor, {"attention_mask": torch.ones(5)}]],
            active_scale=2.0,
            device="cpu",
        )[0][0][0]
        all_zero = node.enhance(
            [[tensor, {"attention_mask": torch.zeros((1, 5))}]],
            active_scale=2.0,
            device="cpu",
        )[0][0][0]

        self.assertTrue(torch.equal(masked[:, :2], tensor[:, :2] * 2.0))
        self.assertTrue(torch.equal(masked[:, 2:], tensor[:, 2:]))
        self.assertTrue(torch.equal(missing, tensor * 2.0))
        self.assertTrue(torch.equal(invalid_rank, tensor * 2.0))
        self.assertTrue(torch.equal(all_zero, tensor * 2.0))

    def test_klein_enhancer_empty_active_range_is_safe(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        tensor = torch.empty((2, 0, 12288), dtype=torch.float16)
        result = node.enhance(
            [[tensor, {}]],
            per_token_whiten=1.0,
            norm_equalize=1.0,
            active_scale=2.0,
            device="cpu",
        )[0][0][0]
        self.assertEqual(result.shape, tensor.shape)
        self.assertEqual(result.dtype, tensor.dtype)

    def test_klein_enhancer_rejects_unsupported_width(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        with self.assertRaisesRegex(ValueError, "expected 12288.*or 7680"):
            node.enhance(
                [[torch.ones((1, 2, 4096)), {}]],
                active_scale=2.0,
                device="cpu",
            )

    def test_klein_enhancer_validates_all_numeric_and_boolean_inputs(self):
        node = ENHANCER.NunchakuKleinEnhancer()
        conditioning = [[torch.ones((1, 1, 12288)), {}]]
        invalid_values = {
            "active_scale": float("nan"),
            "per_token_whiten": 6.0,
            "norm_equalize": -0.1,
            "early_layer_scale": 6.0,
            "mid_layer_scale": -0.1,
            "late_layer_scale": float("inf"),
            "preserve_original": 2.0,
            "active_end_override": 1.5,
            "debug": 1,
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    node.enhance(conditioning, **{name: value})


if __name__ == "__main__":
    unittest.main()

