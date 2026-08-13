import copy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import pathlib
import sys
import unittest

import torch
from torch import nn


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_text_ref_balance_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_text_ref_balance_test.models.klein_wrapper import (
    KleinLoraSpec,
    KleinRefLatentWeightSpec,
    KleinTextRefBalanceSpec,
    LORA_SPEC_OPTION,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_WEIGHT_OPTION,
    TEXT_REF_BALANCE_OPTION,
)
from nunchaku_klein_text_ref_balance_test.nodes.enhancer.common import (
    REFERENCE_CATEGORY,
)


class Result:
    def __init__(self, sample):
        self.sample = sample


class BalanceTransformer(nn.Module):
    """Return constants that identify REF_PROXY, FULL, and TEXT_ONLY routes."""

    def __init__(self):
        super().__init__()
        self.offload = False
        self.calls = 0
        self.fail_on_call = None
        self.routes = []

    def forward(self, *, hidden_states, encoder_hidden_states, img_ids, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("induced prediction failure")
        has_references = bool(torch.count_nonzero(img_ids[:, 0]))
        zero_context = bool(torch.count_nonzero(encoder_hidden_states) == 0)
        self.routes.append(
            {
                "has_references": has_references,
                "zero_context": zero_context,
                "context_shape": tuple(encoder_hidden_states.shape),
                "context_dtype": encoder_hidden_states.dtype,
                "context_device": encoder_hidden_states.device,
                "image_tokens": hidden_states.shape[1],
            }
        )
        if zero_context and has_references:
            value = 10.0
        elif has_references:
            value = 20.0
        elif not zero_context:
            value = 30.0
        else:
            value = -1.0
        return Result(torch.full_like(hidden_states, value))


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


def make_adapter(dtype=torch.float32):
    transformer = BalanceTransformer()
    adapter = NunchakuFlux2KleinAdapter(
        transformer,
        in_channels=2,
        context_dim=4,
        patch_size=1,
        axes_dim=(1, 1, 1, 1),
        dtype=dtype,
        architecture_profile="test",
    )
    return adapter, transformer


def run(adapter, balance=None, *, batch=1, refs=True, context=None, options=None):
    transformer_options = dict(options or {})
    if balance is not None:
        transformer_options[TEXT_REF_BALANCE_OPTION] = KleinTextRefBalanceSpec(
            balance=balance,
            debug=False,
        )
    if context is None:
        context = torch.full((batch, 3, 4), 7.0, dtype=adapter.dtype)
    kwargs = {}
    reference = None
    if refs:
        reference = torch.full((batch, 2, 1, 1), 2.0, dtype=adapter.dtype)
        kwargs = {"ref_latents": [reference], "ref_latents_method": "index"}
    output = adapter(
        torch.zeros((batch, 2, 1, 1), dtype=adapter.dtype),
        torch.ones((batch,), dtype=adapter.dtype),
        context,
        transformer_options=transformer_options,
        **kwargs,
    )
    return output, context, reference


class TextRefBalanceTests(unittest.TestCase):
    def setUp(self):
        self.node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinTextRefBalance"]()
        self.conditioning = [[torch.ones((1, 3, 4)), {"marker": "kept"}]]

    def test_registration_and_exact_original_ui_contract(self):
        cls = type(self.node)
        self.assertEqual(cls.CATEGORY, REFERENCE_CATEGORY)
        self.assertEqual(cls.RETURN_TYPES, ("MODEL", "CONDITIONING"))
        self.assertEqual(cls.FUNCTION, "balance_streams")
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS["NunchakuKleinTextRefBalance"],
            "Nunchaku FLUX.2 Klein Text/Ref Balance",
        )
        inputs = cls.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["model", "conditioning", "balance"])
        self.assertEqual(list(inputs["optional"]), ["debug"])
        self.assertEqual(
            inputs["required"]["balance"],
            ("FLOAT", {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.005}),
        )
        self.assertEqual(inputs["optional"]["debug"], ("BOOLEAN", {"default": False}))

    def test_node_clones_preserves_conditioning_identity_and_attaches_frozen_spec(self):
        adapter, _ = make_adapter()
        source = FakeModelPatcher(adapter)
        branch, conditioning = self.node.balance_streams(
            source, self.conditioning, 0.25, True
        )
        self.assertIsNot(branch, source)
        self.assertIs(branch.model, source.model)
        self.assertEqual(branch.size, source.size)
        self.assertIs(conditioning, self.conditioning)
        self.assertNotIn(TEXT_REF_BALANCE_OPTION, source.model_options["transformer_options"])
        attached = branch.model_options["transformer_options"][TEXT_REF_BALANCE_OPTION]
        self.assertEqual(attached, KleinTextRefBalanceSpec(0.25, True))
        self.assertFalse(any(value is self.conditioning for value in vars(attached).values()))
        with self.assertRaises(FrozenInstanceError):
            attached.balance = 0.5

    def test_node_rejects_second_spec_and_ref_weight_in_both_orders(self):
        adapter, _ = make_adapter()
        source = FakeModelPatcher(adapter)
        balance_branch = self.node.balance_streams(source, self.conditioning, 0.5)[0]
        with self.assertRaisesRegex(ValueError, "Only one active"):
            self.node.balance_streams(balance_branch, self.conditioning, 0.5)

        ref_node = PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinRefLatentWeight"]()
        with self.assertRaisesRegex(ValueError, "cannot be stacked"):
            ref_node.execute(balance_branch, 0, 1.0)
        ref_branch = ref_node.execute(source, 0, 1.0)[0]
        with self.assertRaisesRegex(ValueError, "cannot be stacked"):
            self.node.balance_streams(ref_branch, self.conditioning, 0.5)

    def test_node_validates_balance_debug_and_conditioning(self):
        adapter, _ = make_adapter()
        source = FakeModelPatcher(adapter)
        with self.assertRaises(ValueError):
            self.node.balance_streams(source, self.conditioning, float("nan"))
        with self.assertRaises(TypeError):
            self.node.balance_streams(source, self.conditioning, 0.5, debug=1)
        with self.assertRaises(TypeError):
            self.node.balance_streams(source, object(), 0.5)

    def test_exact_endpoint_routes_and_values(self):
        expected = {
            0.0: (10.0, True, True),
            0.5: (20.0, True, False),
            1.0: (30.0, False, False),
        }
        for balance, (value, has_refs, zero_context) in expected.items():
            with self.subTest(balance=balance):
                adapter, transformer = make_adapter()
                output, _, _ = run(adapter, balance)
                self.assertTrue(torch.equal(output, torch.full_like(output, value)))
                self.assertEqual(transformer.calls, 1)
                self.assertEqual(transformer.routes[0]["has_references"], has_refs)
                self.assertEqual(transformer.routes[0]["zero_context"], zero_context)

    def test_piecewise_residual_arithmetic_is_exact(self):
        for balance, expected, routes in (
            (0.25, 15.0, [(True, True), (True, False)]),
            (0.75, 25.0, [(False, False), (True, False)]),
        ):
            with self.subTest(balance=balance):
                adapter, transformer = make_adapter()
                output, _, _ = run(adapter, balance)
                self.assertTrue(torch.equal(output, torch.full_like(output, expected)))
                self.assertEqual(transformer.calls, 2)
                self.assertEqual(
                    [(route["has_references"], route["zero_context"]) for route in transformer.routes],
                    routes,
                )

    def test_values_near_midpoint_use_the_correct_side(self):
        adapter, transformer = make_adapter()
        low, _, _ = run(adapter, 0.499)
        self.assertEqual(transformer.routes[0]["zero_context"], True)
        self.assertAlmostEqual(float(low.flatten()[0]), 19.98, places=4)
        adapter, transformer = make_adapter()
        high, _, _ = run(adapter, 0.501)
        self.assertEqual(transformer.routes[0]["has_references"], False)
        self.assertAlmostEqual(float(high.flatten()[0]), 20.02, places=4)

    def test_zero_context_contract_and_inputs_are_immutable(self):
        adapter, transformer = make_adapter(dtype=torch.float64)
        context = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) + 1
        context_before = context.clone()
        output, _, reference = run(adapter, 0.25, context=context)
        route = transformer.routes[0]
        self.assertEqual(route["context_shape"], tuple(context.shape))
        self.assertEqual(route["context_dtype"], context.dtype)
        self.assertEqual(route["context_device"], context.device)
        self.assertTrue(torch.equal(context, context_before))
        self.assertTrue(torch.equal(reference, torch.full_like(reference, 2.0)))
        self.assertEqual(output.dtype, torch.float64)

    def test_requires_runtime_references(self):
        adapter, _ = make_adapter()
        with self.assertRaisesRegex(ValueError, "requires a non-empty runtime reference"):
            run(adapter, 0.5, refs=False)

    def test_no_spec_preserves_ordinary_t2i_and_reference_paths(self):
        adapter, transformer = make_adapter()
        t2i, _, _ = run(adapter, refs=False)
        reference, _, _ = run(adapter, refs=True)
        self.assertTrue(torch.equal(t2i, torch.full_like(t2i, 30.0)))
        self.assertTrue(torch.equal(reference, torch.full_like(reference, 20.0)))
        self.assertEqual(transformer.calls, 2)

    def test_serialized_batch_call_counts(self):
        for balance, calls_per_item in ((0.0, 1), (0.5, 1), (1.0, 1), (0.25, 2), (0.75, 2)):
            with self.subTest(balance=balance):
                adapter, transformer = make_adapter()
                run(adapter, balance, batch=2)
                self.assertEqual(transformer.calls, 2 * calls_per_item)

    def test_active_lora_identity_is_stable_across_both_predictions(self):
        adapter, _ = make_adapter()
        lora = KleinLoraSpec(Path("A.safetensors"), 1, 2, 1.0, {})
        adapter.shared_lora_state.active_identity = (lora.identity,)
        output, _, _ = run(
            adapter,
            0.25,
            options={LORA_SPEC_OPTION: (lora,)},
        )
        self.assertTrue(torch.equal(output, torch.full_like(output, 15.0)))
        self.assertEqual(adapter.shared_lora_state.active_identity, (lora.identity,))
        self.assertEqual(adapter.shared_lora_state.generation, 0)

    def test_prediction_failure_retains_no_state_and_does_not_corrupt_lora_identity(self):
        for failing_call in (1, 2):
            with self.subTest(failing_call=failing_call):
                adapter, transformer = make_adapter()
                lora = KleinLoraSpec(Path("A.safetensors"), 1, 2, 1.0, {})
                adapter.shared_lora_state.active_identity = (lora.identity,)
                transformer.fail_on_call = failing_call
                with self.assertRaisesRegex(RuntimeError, "induced prediction failure"):
                    run(adapter, 0.25, options={LORA_SPEC_OPTION: (lora,)})
                self.assertEqual(adapter.shared_lora_state.active_identity, (lora.identity,))
                self.assertFalse(any(name in vars(adapter) for name in ("prediction", "zero_context", "references")))


if __name__ == "__main__":
    unittest.main()
