import copy
from dataclasses import FrozenInstanceError
import importlib.util
import pathlib
import sys
import unittest

import torch
from torch import nn


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_ref_weight_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_ref_weight_test.models.klein_wrapper import (
    KleinRefLatentWeightSpec,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_WEIGHT_OPTION,
)
from nunchaku_klein_ref_weight_test.nodes.enhancer.common import REFERENCE_CATEGORY


class Result:
    def __init__(self, sample):
        self.sample = sample


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offload = False
        self.calls = 0
        self.layouts = []
        self.fail_on_call = None

    def forward(self, *, hidden_states, img_ids, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("induced transformer failure")
        self.layouts.append(
            (
                tuple(img_ids[:, 0].tolist()),
                tuple(hidden_states[0, 1:, 0].tolist()),
            )
        )
        reference_sum = hidden_states[:, 1:].sum(dim=1, keepdim=True)
        return Result(reference_sum.expand_as(hidden_states).clone())


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
    transformer = FakeTransformer()
    adapter = NunchakuFlux2KleinAdapter(
        transformer,
        in_channels=2,
        context_dim=4,
        patch_size=1,
        axes_dim=(1, 1, 1, 1),
        dtype=torch.float32,
    )
    return adapter, transformer


def run(adapter, refs=None, *, batch=1, spec=None):
    options = {}
    if spec is not None:
        options[REF_LATENT_WEIGHT_OPTION] = spec
    kwargs = {}
    if refs is not None:
        kwargs["ref_latents"] = [ref.expand(batch, -1, -1, -1) for ref in refs]
        kwargs["ref_latents_method"] = "index"
    return adapter(
        torch.zeros((batch, 2, 1, 1)),
        torch.ones((batch,)),
        torch.zeros((batch, 1, 4)),
        transformer_options=options,
        **kwargs,
    )


class RefLatentWeightTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeight"]()
        self.a = torch.full((1, 2, 1, 1), 2.0)
        self.b = torch.full((1, 2, 1, 1), 4.0)

    def test_registration_and_exact_original_ui_contract(self):
        cls = type(self.node)
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL",))
        self.assertEqual(cls.FUNCTION, "execute")
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS["NunchakuKleinRefLatentWeight"],
            "Nunchaku FLUX.2 Klein Ref Latent Weight",
        )
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "reference_index", "weight"])
        self.assertEqual(required["reference_index"], ("INT", {"default": 0, "min": 0, "max": 7}))
        self.assertEqual(required["weight"], ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}))

    def test_node_clones_and_attaches_immutable_spec(self):
        adapter, _ = make_adapter()
        source = FakeModelPatcher(adapter)
        marker = "existing-lora-spec"
        source.model_options["transformer_options"]["lora"] = marker
        branch = self.node.execute(source, 1, 0.5)[0]
        self.assertIsNot(branch, source)
        self.assertNotIn(REF_LATENT_WEIGHT_OPTION, source.model_options["transformer_options"])
        attached = branch.model_options["transformer_options"][REF_LATENT_WEIGHT_OPTION]
        self.assertEqual(attached, KleinRefLatentWeightSpec(1, 0.5))
        self.assertEqual(branch.model_options["transformer_options"]["lora"], marker)
        with self.assertRaises(FrozenInstanceError):
            attached.weight = 2.0
        with self.assertRaisesRegex(ValueError, "Only one active"):
            self.node.execute(branch, 0, 1.0)

    def test_exact_endpoints_and_single_reference_ablation(self):
        adapter, transformer = make_adapter()
        full = run(adapter, [self.a], spec=KleinRefLatentWeightSpec(0, 1.0))
        self.assertTrue(torch.equal(full, torch.full_like(full, 2.0)))
        self.assertEqual(transformer.calls, 1)
        transformer.calls = 0
        omitted = run(adapter, [self.a], spec=KleinRefLatentWeightSpec(0, 0.0))
        self.assertTrue(torch.equal(omitted, torch.zeros_like(omitted)))
        self.assertEqual(transformer.calls, 1)

    def test_residual_arithmetic_and_extrapolation_are_exact(self):
        adapter, transformer = make_adapter()
        half = run(adapter, [self.a, self.b], spec=KleinRefLatentWeightSpec(0, 0.5))
        self.assertTrue(torch.equal(half, torch.full_like(half, 5.0)))
        self.assertEqual(transformer.calls, 2)
        transformer.calls = 0
        doubled = run(adapter, [self.a, self.b], spec=KleinRefLatentWeightSpec(0, 2.0))
        self.assertTrue(torch.equal(doubled, torch.full_like(doubled, 8.0)))
        self.assertEqual(transformer.calls, 2)

    def test_omission_preserves_remaining_order_and_input_tensors(self):
        adapter, transformer = make_adapter()
        c = torch.full_like(self.a, 8.0)
        snapshots = [tensor.clone() for tensor in (self.a, self.b, c)]
        run(adapter, [self.a, self.b, c], spec=KleinRefLatentWeightSpec(1, 0.0))
        self.assertEqual(transformer.layouts[-1][0], (0.0, 10.0, 20.0))
        self.assertEqual(transformer.layouts[-1][1], (2.0, 8.0))
        for current, snapshot in zip((self.a, self.b, c), snapshots, strict=True):
            self.assertTrue(torch.equal(current, snapshot))

    def test_no_spec_preserves_t2i_and_reference_paths(self):
        adapter, transformer = make_adapter()
        self.assertTrue(torch.equal(run(adapter), torch.zeros((1, 2, 1, 1))))
        self.assertTrue(torch.equal(run(adapter, [self.a, self.b]), torch.full((1, 2, 1, 1), 6.0)))
        self.assertEqual(transformer.calls, 2)

    def test_invalid_runtime_reference_index_is_actionable(self):
        adapter, _ = make_adapter()
        with self.assertRaisesRegex(IndexError, "out of range for 1 runtime references"):
            run(adapter, [self.a], spec=KleinRefLatentWeightSpec(1, 0.5))
        with self.assertRaisesRegex(ValueError, "requires a non-empty runtime reference"):
            run(adapter, spec=KleinRefLatentWeightSpec(0, 0.5))

    def test_batch_serialization_scales_residual_call_count(self):
        adapter, transformer = make_adapter()
        run(adapter, [self.a], batch=2, spec=KleinRefLatentWeightSpec(0, 1.0))
        self.assertEqual(transformer.calls, 2)
        transformer.calls = 0
        run(adapter, [self.a], batch=2, spec=KleinRefLatentWeightSpec(0, 0.5))
        self.assertEqual(transformer.calls, 4)

    def test_failed_second_pass_retains_no_adapter_prediction_state(self):
        adapter, transformer = make_adapter()
        transformer.fail_on_call = 2
        with self.assertRaisesRegex(RuntimeError, "induced transformer failure"):
            run(adapter, [self.a], spec=KleinRefLatentWeightSpec(0, 0.5))
        transformer.fail_on_call = None
        output = run(adapter, [self.a], spec=KleinRefLatentWeightSpec(0, 1.0))
        self.assertTrue(torch.equal(output, torch.full_like(output, 2.0)))
        self.assertFalse(any("prediction" in name for name in vars(adapter)))


if __name__ == "__main__":
    unittest.main()
