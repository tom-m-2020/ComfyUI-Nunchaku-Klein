import gc
import importlib.util
import pathlib
import sys
import unittest
import weakref

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_color_anchor_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_color_anchor_test.nodes import enhancer as ENHANCER
from nunchaku_klein_color_anchor_test.nodes.enhancer.common import POST_CFG_CATEGORY
from nunchaku_klein_color_anchor_test.nodes.enhancer.post_cfg import POST_CFG_OPTION


class FakeModelPatcher:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})
        self.clone_source = None

    def clone(self):
        branch = FakeModelPatcher(self.model_options)
        branch.clone_source = self
        return branch


def conditioning(reference=None):
    metadata = {"marker": object()}
    if reference is not None:
        metadata["reference_latents"] = [reference]
    return [[torch.zeros((1, 2, 4)), metadata]]


def invoke(callback, denoised, sigma, model_options=None):
    return callback(
        {
            "denoised": denoised,
            "cond": [],
            "uncond": [],
            "cond_scale": 1.0,
            "model": object(),
            "uncond_denoised": denoised,
            "cond_denoised": denoised,
            "sigma": torch.tensor([sigma], device=denoised.device),
            "model_options": model_options if model_options is not None else {},
            "input": denoised,
        }
    )


class ColorAnchorTests(unittest.TestCase):
    def setUp(self):
        self.node = ENHANCER.NunchakuKleinColorAnchor()

    def callback(self, model):
        return model.model_options[POST_CFG_OPTION][-1]

    def test_registration_category_and_exact_ui_contract(self):
        node_id = "NunchakuKleinColorAnchor"
        self.assertIs(PACKAGE.NODE_CLASS_MAPPINGS[node_id], type(self.node))
        self.assertEqual(type(self.node).CATEGORY, POST_CFG_CATEGORY)
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[node_id],
            "Nunchaku FLUX.2 Klein Color Anchor",
        )
        self.assertEqual(type(self.node).RETURN_TYPES, ("MODEL",))
        inputs = type(self.node).INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["model", "conditioning", "strength"])
        self.assertEqual(
            list(inputs["optional"]),
            ["ramp_curve", "ref_index", "channel_weights", "debug"],
        )
        self.assertEqual(
            inputs["required"]["strength"][1],
            {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "tooltip": (
                    "Maximum correction strength. "
                    "0.3-0.6 is a good starting range. "
                    "Too high and you override the model's color decisions entirely."
                ),
            },
        )
        self.assertEqual(inputs["optional"]["ramp_curve"][1]["default"], 1.5)
        self.assertEqual(inputs["optional"]["ramp_curve"][1]["step"], 0.1)
        self.assertEqual(inputs["optional"]["ref_index"][1], {
            "default": 0,
            "min": 0,
            "max": 63,
            "tooltip": "Which reference latent to anchor colors from.",
        })
        self.assertEqual(inputs["optional"]["channel_weights"][0], ["uniform", "by_variance"])

    def test_strength_zero_is_true_identity(self):
        model = FakeModelPatcher()
        output = self.node.apply(model, object(), strength=0.0)[0]
        self.assertIs(output, model)
        self.assertNotIn(POST_CFG_OPTION, model.model_options)

    def test_clone_preserves_existing_callbacks_and_source(self):
        calls = []

        def first(args):
            calls.append("first")
            return args["denoised"] + 1

        model = FakeModelPatcher({POST_CFG_OPTION: [first], "marker": "kept"})
        reference = torch.full((1, 2, 2, 2), 3.0)
        branch = self.node.apply(
            model, conditioning(reference), strength=1.0, ramp_curve=1.0
        )[0]
        self.assertIsNot(branch, model)
        self.assertIs(branch.clone_source, model)
        self.assertEqual(model.model_options[POST_CFG_OPTION], [first])
        self.assertEqual(branch.model_options[POST_CFG_OPTION][0], first)
        self.assertEqual(len(branch.model_options[POST_CFG_OPTION]), 2)
        self.assertEqual(branch.model_options["marker"], "kept")
        current = torch.zeros((1, 2, 2, 2))
        for callback in branch.model_options[POST_CFG_OPTION]:
            current = invoke(callback, current, 1.0)
        self.assertEqual(calls, ["first"])
        self.assertTrue(torch.allclose(current.mean((-2, -1)), torch.full((1, 2), 2.0)))

    def test_missing_and_out_of_range_reference_leave_model_unchanged(self):
        model = FakeModelPatcher()
        self.assertIs(self.node.apply(model, conditioning(), strength=0.5)[0], model)
        reference = torch.ones((1, 2, 2, 2))
        self.assertIs(
            self.node.apply(
                model, conditioning(reference), strength=0.5, ref_index=1
            )[0],
            model,
        )

    def test_uniform_corrects_means_and_preserves_spatial_deviations(self):
        reference = torch.tensor(
            [[[[2.0, 2.0], [2.0, 2.0]], [[-1.0, -1.0], [-1.0, -1.0]]]]
        )
        branch = self.node.apply(
            FakeModelPatcher(), conditioning(reference), strength=1.0, ramp_curve=1.0
        )[0]
        denoised = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 3.0], [5.0, 7.0]]]],
            dtype=torch.float64,
        )
        before_deviation = denoised - denoised.mean((-2, -1), keepdim=True)
        output = invoke(self.callback(branch), denoised, 1.0)
        expected_mean = denoised.mean((-2, -1)) + 0.5 * (
            torch.tensor([[2.0, -1.0]], dtype=denoised.dtype)
            - denoised.mean((-2, -1))
        )
        self.assertTrue(torch.allclose(output.mean((-2, -1)), expected_mean))
        self.assertTrue(
            torch.allclose(
                output - output.mean((-2, -1), keepdim=True), before_deviation
            )
        )
        self.assertEqual(output.dtype, denoised.dtype)
        self.assertEqual(output.device, denoised.device)

    def test_by_variance_matches_original_weighting(self):
        reference = torch.tensor(
            [[[[2.0, 2.0], [2.0, 2.0]], [[0.0, 2.0], [0.0, 2.0]]]]
        )
        branch = self.node.apply(
            FakeModelPatcher(),
            conditioning(reference),
            strength=1.0,
            ramp_curve=1.0,
            channel_weights="by_variance",
        )[0]
        denoised = torch.zeros((1, 2, 2, 2))
        output = invoke(self.callback(branch), denoised, 1.0)
        variance = reference.float().var((-2, -1), keepdim=True)
        trust = 1.0 / (1.0 + variance)
        trust = trust / trust.max().clamp(min=1e-8)
        expected = reference.float().mean((-2, -1), keepdim=True) * trust * 0.5
        self.assertTrue(torch.allclose(output.mean((-2, -1), keepdim=True), expected))

    def test_schedule_start_middle_end_and_curves(self):
        reference = torch.ones((1, 1, 2, 2))
        for curve, first_effective in ((1.0, 0.5), (2.0, 0.5**0.5), (0.5, 0.25)):
            branch = self.node.apply(
                FakeModelPatcher(),
                conditioning(reference),
                strength=1.0,
                ramp_curve=curve,
            )[0]
            callback = self.callback(branch)
            model_options = {}
            start = invoke(callback, torch.zeros_like(reference), 1.0, model_options)
            middle = invoke(callback, torch.zeros_like(reference), 0.5, model_options)
            end = invoke(callback, torch.zeros_like(reference), 0.0, model_options)
            self.assertAlmostEqual(float(start.mean()), first_effective, places=6)
            self.assertAlmostEqual(float(middle.mean()), 0.75 ** (1.0 / curve), places=6)
            self.assertAlmostEqual(float(end.mean()), 1.0, places=6)

    def test_cached_branch_resets_progress_for_new_sampler_options(self):
        reference = torch.ones((1, 1, 2, 2))
        branch = self.node.apply(
            FakeModelPatcher(),
            conditioning(reference),
            strength=1.0,
            ramp_curve=1.0,
        )[0]
        callback = self.callback(branch)
        first_run = {}
        invoke(callback, torch.zeros_like(reference), 1.0, first_run)
        invoke(callback, torch.zeros_like(reference), 0.0, first_run)
        second_start = invoke(callback, torch.zeros_like(reference), 1.0, {})
        self.assertAlmostEqual(float(second_start.mean()), 0.5, places=6)

    def test_reference_and_conditioning_are_not_mutated_and_batch_broadcasts(self):
        reference = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        snapshot = reference.clone()
        source_conditioning = conditioning(reference)
        metadata = source_conditioning[0][1]
        branch = self.node.apply(
            FakeModelPatcher(), source_conditioning, strength=1.0
        )[0]
        denoised = torch.zeros((3, 2, 4, 4), dtype=torch.float16)
        output = invoke(self.callback(branch), denoised, 1.0)
        self.assertEqual(output.shape, denoised.shape)
        self.assertEqual(output.dtype, denoised.dtype)
        self.assertTrue(torch.equal(reference, snapshot))
        self.assertIs(source_conditioning[0][1], metadata)
        self.assertIs(source_conditioning[0][1]["reference_latents"][0], reference)

    def test_callback_closure_retains_no_tensor_and_outputs_are_collectable(self):
        reference = torch.ones((1, 2, 2, 2))
        source_conditioning = conditioning(reference)
        branch = self.node.apply(
            FakeModelPatcher(), source_conditioning, strength=1.0
        )[0]
        callback = self.callback(branch)
        closure_values = [cell.cell_contents for cell in callback.__closure__]
        self.assertFalse(any(torch.is_tensor(value) for value in closure_values))

        denoised = torch.zeros((1, 2, 2, 2))
        output = invoke(callback, denoised, 1.0)
        output_ref = weakref.ref(output)
        reference_ref = weakref.ref(reference)
        del output, denoised, source_conditioning, reference
        gc.collect()
        self.assertIsNone(output_ref())
        self.assertIsNone(reference_ref())

    def test_stacked_anchors_append_and_execute_deterministically(self):
        first_reference = torch.ones((1, 1, 2, 2))
        second_reference = torch.full((1, 1, 2, 2), 3.0)
        base = FakeModelPatcher()
        first = self.node.apply(
            base, conditioning(first_reference), strength=1.0, ramp_curve=1.0
        )[0]
        second = self.node.apply(
            first, conditioning(second_reference), strength=1.0, ramp_curve=1.0
        )[0]
        self.assertEqual(len(first.model_options[POST_CFG_OPTION]), 1)
        self.assertEqual(len(second.model_options[POST_CFG_OPTION]), 2)
        value = torch.zeros((1, 1, 2, 2))
        for callback in second.model_options[POST_CFG_OPTION]:
            value = invoke(callback, value, 1.0)
        self.assertAlmostEqual(float(value.mean()), 1.75)

    def test_malformed_node_inputs_fail_closed(self):
        model = FakeModelPatcher()
        reference = torch.ones((1, 2, 2, 2))
        valid = conditioning(reference)
        cases = (
            ({"strength": -0.1}, ValueError, "strength"),
            ({"strength": 1.1}, ValueError, "strength"),
            ({"ramp_curve": 0.4}, ValueError, "ramp_curve"),
            ({"ref_index": 64}, ValueError, "ref_index"),
            ({"channel_weights": "invalid"}, ValueError, "channel_weights"),
            ({"debug": 1}, TypeError, "debug"),
        )
        for kwargs, error, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    self.node.apply(model, valid, **kwargs)

        with self.assertRaisesRegex(TypeError, "conditioning"):
            self.node.apply(model, None)
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            self.node.apply(model, conditioning(object()))
        with self.assertRaisesRegex(ValueError, "BCHW"):
            self.node.apply(model, conditioning(torch.ones((2, 2, 2))))
        with self.assertRaisesRegex(TypeError, "clone"):
            self.node.apply(object(), valid)
        with self.assertRaisesRegex(TypeError, POST_CFG_OPTION):
            self.node.apply(
                FakeModelPatcher({POST_CFG_OPTION: object()}), valid
            )

    def test_malformed_callback_inputs_fail_closed(self):
        reference = torch.ones((2, 2, 2, 2))
        branch = self.node.apply(
            FakeModelPatcher(), conditioning(reference), strength=1.0
        )[0]
        callback = self.callback(branch)
        valid = {
            "denoised": torch.zeros((2, 2, 2, 2)),
            "sigma": torch.ones((1,)),
            "model_options": {},
        }
        with self.assertRaisesRegex(TypeError, "arguments must be a mapping"):
            callback(None)
        for key, value, error, message in (
            ("denoised", torch.zeros((2, 2, 2)), ValueError, "BCHW"),
            ("sigma", torch.empty((0,)), TypeError, "non-empty"),
            ("model_options", object(), TypeError, "mapping"),
        ):
            args = dict(valid)
            args[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(error, message):
                    callback(args)

        args = dict(valid)
        args["denoised"] = torch.zeros((2, 3, 2, 2))
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            callback(args)
        args["denoised"] = torch.zeros((3, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "reference batch"):
            callback(args)


if __name__ == "__main__":
    unittest.main()
