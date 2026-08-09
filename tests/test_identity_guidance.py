import importlib.util
import pathlib
import sys
import unittest

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_identity_guidance_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_identity_guidance_test.nodes import enhancer as ENHANCER
from nunchaku_klein_identity_guidance_test.nodes.enhancer.common import POST_CFG_CATEGORY
from nunchaku_klein_identity_guidance_test.nodes.enhancer.post_cfg import POST_CFG_OPTION


class FakeModelPatcher:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def clone(self):
        return FakeModelPatcher(self.model_options)


def invoke(callback, denoised, sigma):
    return callback({"denoised": denoised, "sigma": torch.tensor([sigma])})


class IdentityGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.node = ENHANCER.NunchakuKleinIdentityGuidance()

    def apply(self, reference, **kwargs):
        return self.node.apply(
            FakeModelPatcher(), {"samples": reference}, **kwargs
        )[0]

    def callback(self, branch):
        return branch.model_options[POST_CFG_OPTION][-1]

    def test_registration_and_exact_ui(self):
        node_id = "NunchakuKleinIdentityGuidance"
        self.assertIs(PACKAGE.NODE_CLASS_MAPPINGS[node_id], type(self.node))
        self.assertEqual(type(self.node).CATEGORY, POST_CFG_CATEGORY)
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[node_id],
            "Nunchaku FLUX.2 Klein Identity Guidance",
        )
        inputs = type(self.node).INPUT_TYPES()["required"]
        self.assertEqual(
            list(inputs),
            ["model", "identity_latent", "strength", "start_percent", "end_percent", "mode"],
        )
        self.assertEqual(inputs["strength"][1]["default"], 0.3)
        self.assertEqual(inputs["start_percent"][1]["step"], 0.05)
        self.assertEqual(inputs["end_percent"][1]["default"], 0.8)
        self.assertEqual(inputs["mode"][0], ["adaptive", "direct", "channel_match"])

    def test_zero_strength_returns_identical_model(self):
        model = FakeModelPatcher()
        self.assertIs(self.node.apply(model, object(), strength=0.0)[0], model)

    def test_direct_mode_and_schedule(self):
        reference = torch.full((1, 2, 2, 2), 4.0)
        callback = self.callback(self.apply(reference, strength=0.25, mode="direct"))
        source = torch.zeros_like(reference)
        self.assertTrue(torch.equal(invoke(callback, source, 1.0), torch.ones_like(source)))
        self.assertTrue(torch.equal(invoke(callback, source, 0.5), torch.ones_like(source)))
        self.assertTrue(torch.equal(invoke(callback, source, 0.1), source))

    def test_adaptive_matches_original_formula(self):
        reference = torch.tensor([[[[2.0]], [[0.0]]]])
        denoised = torch.tensor([[[[1.0]], [[1.0]]]])
        callback = self.callback(self.apply(reference, strength=0.5, mode="adaptive"))
        similarity = torch.nn.functional.cosine_similarity(
            denoised.flatten(2), reference.flatten(2), dim=1
        ).clamp(0, 1).view(1, 1, 1, 1)
        expected = denoised + (reference - denoised) * similarity * 0.5
        self.assertTrue(torch.allclose(invoke(callback, denoised, 0.5), expected))

    def test_channel_match_matches_original_formula(self):
        reference = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
        denoised = torch.tensor([[[[1.0, 2.0], [4.0, 7.0]]]])
        callback = self.callback(self.apply(reference, strength=0.4, mode="channel_match"))
        matched = (
            (denoised - denoised.mean((2, 3), keepdim=True))
            / denoised.std((2, 3), keepdim=True).clamp(min=1e-5)
            * reference.std((2, 3), keepdim=True).clamp(min=1e-5)
            + reference.mean((2, 3), keepdim=True)
        )
        self.assertTrue(
            torch.allclose(invoke(callback, denoised, 0.5), denoised + (matched - denoised) * 0.4)
        )

    def test_callback_order_and_reference_is_private_cpu_copy(self):
        calls = []
        def first(args):
            calls.append("first")
            return args["denoised"] + 1
        model = FakeModelPatcher({POST_CFG_OPTION: [first]})
        reference = torch.full((1, 1, 2, 2), 4.0)
        branch = self.node.apply(
            model, {"samples": reference}, strength=0.5, mode="direct"
        )[0]
        reference.zero_()
        value = torch.zeros((1, 1, 2, 2))
        for callback in branch.model_options[POST_CFG_OPTION]:
            value = invoke(callback, value, 0.5)
        self.assertEqual(calls, ["first"])
        self.assertTrue(torch.equal(value, torch.full_like(value, 2.5)))
        self.assertEqual(len(model.model_options[POST_CFG_OPTION]), 1)

    def test_resize_batch_channel_dtype_and_input_immutability(self):
        reference = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        snapshot = reference.clone()
        callback = self.callback(self.apply(reference, strength=1.0, mode="direct"))
        denoised = torch.zeros((2, 2, 4, 4), dtype=torch.float16)
        output = invoke(callback, denoised, 0.5)
        self.assertEqual(output.shape, denoised.shape)
        self.assertEqual(output.dtype, denoised.dtype)
        self.assertTrue(torch.equal(output[:, 0], torch.ones_like(output[:, 0])))
        self.assertTrue(torch.equal(output[:, 1], torch.zeros_like(output[:, 1])))
        self.assertTrue(torch.equal(reference, snapshot))
        self.assertTrue(torch.equal(denoised, torch.zeros_like(denoised)))


if __name__ == "__main__":
    unittest.main()
