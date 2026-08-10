import importlib.util
import pathlib
import sys
import types
import unittest

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "klein_lora_normalization_test_module",
    TARGET / "nodes" / "klein_lora_normalization.py",
)
normalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalization)
normalize_klein_lora_state = normalization.normalize_klein_lora_state

wrapper_spec = importlib.util.spec_from_file_location(
    "klein_lora_wrapper_test_module",
    TARGET / "models" / "klein_wrapper.py",
)
wrapper = importlib.util.module_from_spec(wrapper_spec)
sys.modules[wrapper_spec.name] = wrapper
wrapper_spec.loader.exec_module(wrapper)


def add_pair(state, target, rank, in_features, out_features, *, dtype=torch.float64):
    state[f"{target}.lora.down.weight"] = torch.randn(
        rank, in_features, dtype=dtype
    )
    state[f"{target}.lora.up.weight"] = torch.randn(
        out_features, rank, dtype=dtype
    )


def add_ab_pair(state, target, rank, in_features, out_features):
    state[f"{target}.lora_A.weight"] = torch.randn(rank, in_features)
    state[f"{target}.lora_B.weight"] = torch.randn(out_features, rank)


class KleinLoraNormalizationTests(unittest.TestCase):
    def test_canonical_state_is_validated_and_returned_unchanged(self):
        state = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": (
                torch.randn(5, 7)
            ),
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": (
                torch.randn(11, 5)
            ),
        }

        normalized = normalize_klein_lora_state(state)

        self.assertIs(normalized, state)

    def test_arbitrary_qkv_ranks_are_fused_exactly_without_source_mutation(self):
        state = {}
        prefix = "transformer.transformer_blocks.2.attn"
        ranks = (2, 3, 4)
        names = ("to_q", "to_k", "to_v")
        for name, rank in zip(names, ranks):
            add_pair(state, f"{prefix}.{name}", rank, 7, 5)
        before = {key: value.clone() for key, value in state.items()}

        normalized = normalize_klein_lora_state(state)
        fused_a = normalized[
            "diffusion_model.double_blocks.2.img_attn.qkv.lora_A.weight"
        ]
        fused_b = normalized[
            "diffusion_model.double_blocks.2.img_attn.qkv.lora_B.weight"
        ]
        expected = torch.cat(
            [
                state[f"{prefix}.{name}.lora.up.weight"]
                @ state[f"{prefix}.{name}.lora.down.weight"]
                for name in names
            ],
            dim=0,
        )

        self.assertEqual(list(fused_a.shape), [sum(ranks), 7])
        self.assertEqual(list(fused_b.shape), [15, sum(ranks)])
        self.assertTrue(torch.allclose(fused_b @ fused_a, expected, rtol=0, atol=1e-12))
        self.assertEqual(fused_a.dtype, torch.float64)
        self.assertEqual(fused_a.device.type, "cpu")
        self.assertEqual(set(state), set(before))
        for key in state:
            self.assertTrue(torch.equal(state[key], before[key]))

    def test_text_qkv_and_all_direct_targets_map_to_bfl_names(self):
        state = {}
        double = "transformer.transformer_blocks.1"
        for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
            add_ab_pair(state, f"{double}.attn.{name}", 2, 4, 3)
        direct = {
            "attn.to_out.0": "img_attn.proj",
            "attn.to_add_out": "txt_attn.proj",
            "ff.linear_in": "img_mlp.0",
            "ff.linear_out": "img_mlp.2",
            "ff_context.linear_in": "txt_mlp.0",
            "ff_context.linear_out": "txt_mlp.2",
        }
        for source in direct:
            if source == "attn.to_out.0":
                add_pair(state, f"{double}.{source}", 2, 4, 6)
            else:
                add_ab_pair(state, f"{double}.{source}", 2, 4, 6)
        single = "transformer.single_transformer_blocks.3.attn"
        add_ab_pair(state, f"{single}.to_qkv_mlp_proj", 2, 4, 9)
        add_ab_pair(state, f"{single}.to_out", 2, 9, 4)

        normalized = normalize_klein_lora_state(state)

        self.assertIn(
            "diffusion_model.double_blocks.1.txt_attn.qkv.lora_A.weight",
            normalized,
        )
        for canonical in direct.values():
            source = next(key for key, value in direct.items() if value == canonical)
            destination = f"diffusion_model.double_blocks.1.{canonical}"
            suffixes = (
                ("lora.down.weight", "lora.up.weight")
                if source == "attn.to_out.0"
                else ("lora_A.weight", "lora_B.weight")
            )
            self.assertIs(
                normalized[f"{destination}.lora_A.weight"],
                state[f"{double}.{source}.{suffixes[0]}"],
            )
            self.assertIs(
                normalized[f"{destination}.lora_B.weight"],
                state[f"{double}.{source}.{suffixes[1]}"],
            )
        self.assertIn(
            "diffusion_model.single_blocks.3.linear1.lora_A.weight", normalized
        )
        self.assertIn(
            "diffusion_model.single_blocks.3.linear2.lora_B.weight", normalized
        )
        self.assertEqual(len(normalized), 18)

    def test_incomplete_qkv_group_fails_with_missing_targets(self):
        state = {}
        prefix = "transformer.transformer_blocks.0.attn"
        add_pair(state, f"{prefix}.to_q", 2, 4, 3)
        add_pair(state, f"{prefix}.to_k", 2, 4, 3)

        with self.assertRaisesRegex(ValueError, "missing targets.*to_v"):
            normalize_klein_lora_state(state)

    def test_incomplete_pair_fails_with_missing_key(self):
        state = {
            "transformer.transformer_blocks.0.ff.linear_in.lora.down.weight": (
                torch.randn(2, 4)
            )
        }

        with self.assertRaisesRegex(ValueError, "missing keys.*lora.up.weight"):
            normalize_klein_lora_state(state)

    def test_pair_rank_mismatch_fails(self):
        target = "transformer.transformer_blocks.0.ff.linear_in"
        state = {
            f"{target}.lora.down.weight": torch.randn(2, 4),
            f"{target}.lora.up.weight": torch.randn(6, 3),
        }

        with self.assertRaisesRegex(ValueError, "inconsistent rank"):
            normalize_klein_lora_state(state)

    def test_pair_dtype_mismatch_fails(self):
        target = "transformer.transformer_blocks.0.ff.linear_in"
        state = {
            f"{target}.lora_A.weight": torch.randn(2, 4, dtype=torch.float32),
            f"{target}.lora_B.weight": torch.randn(6, 2, dtype=torch.float64),
        }

        with self.assertRaisesRegex(ValueError, "mismatched dtypes"):
            normalize_klein_lora_state(state)

    def test_qkv_input_dimension_mismatch_fails(self):
        state = {}
        prefix = "transformer.transformer_blocks.0.attn"
        add_pair(state, f"{prefix}.to_q", 2, 4, 3)
        add_pair(state, f"{prefix}.to_k", 2, 5, 3)
        add_pair(state, f"{prefix}.to_v", 2, 4, 3)

        with self.assertRaisesRegex(ValueError, "inputs differ"):
            normalize_klein_lora_state(state)

    def test_unknown_and_mixed_layouts_fail_without_partial_output(self):
        state = {}
        add_pair(
            state,
            "transformer.transformer_blocks.0.ff.linear_in",
            2,
            4,
            6,
        )
        state[
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight"
        ] = torch.randn(2, 4)
        state[
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight"
        ] = torch.randn(6, 2)

        with self.assertRaisesRegex(ValueError, "Unsupported or mixed"):
            normalize_klein_lora_state(state)

    def test_composition_receives_canonical_states_in_graph_order(self):
        first_state = {
            "diffusion_model.single_blocks.0.linear1.lora_A.weight": torch.randn(2, 3),
            "diffusion_model.single_blocks.0.linear1.lora_B.weight": torch.randn(4, 2),
        }
        second_raw = {}
        add_pair(
            second_raw,
            "transformer.single_transformer_blocks.1.attn.to_out",
            3,
            5,
            6,
        )
        second_state = normalize_klein_lora_state(second_raw)
        first = wrapper.KleinLoraSpec(
            pathlib.Path("first.safetensors"), 10, 20, -0.5, first_state
        )
        second = wrapper.KleinLoraSpec(
            pathlib.Path("second.safetensors"), 30, 40, 2.0, second_state
        )
        calls = []

        common = types.ModuleType("nunchaku.lora.common")
        common.compose_lora = lambda entries: calls.append(entries) or {
            "prepared": torch.ones(1)
        }
        lora = types.ModuleType("nunchaku.lora")
        nunchaku = types.ModuleType("nunchaku")
        saved = {
            name: sys.modules.get(name)
            for name in ("nunchaku", "nunchaku.lora", "nunchaku.lora.common")
        }
        sys.modules.update(
            {
                "nunchaku": nunchaku,
                "nunchaku.lora": lora,
                "nunchaku.lora.common": common,
            }
        )
        try:
            wrapper.KleinLoraPreparationCache().prepare((first, second))
        finally:
            for name, previous in saved.items():
                if previous is None:
                    del sys.modules[name]
                else:
                    sys.modules[name] = previous

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][0], first_state)
        self.assertEqual(calls[0][0][1], -0.5)
        self.assertIs(calls[0][1][0], second_state)
        self.assertEqual(calls[0][1][1], 2.0)
        self.assertEqual(first.source_identity, ("lora", "first.safetensors", 10, 20))
        self.assertEqual(first.identity, (*first.source_identity, -0.5))


if __name__ == "__main__":
    unittest.main()
