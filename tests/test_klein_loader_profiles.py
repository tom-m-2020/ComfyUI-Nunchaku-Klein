import importlib.util
import json
import pathlib
import sys
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_loader_profiles_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_loader_profiles_test.nodes.klein_loader import (
    KLEIN_ARCHITECTURE_PROFILES,
    _validated_comfy_config,
)
from nunchaku_klein_loader_profiles_test.nodes.klein_lora import (
    _lora_cache_identity,
    _validate_lora_profile,
)


def metadata(*, context, heads, double, single):
    config = {
        "_class_name": "Flux2Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [32, 32, 32, 32],
        "eps": 1e-6,
        "guidance_embeds": False,
        "in_channels": 128,
        "joint_attention_dim": context,
        "mlp_ratio": 3.0,
        "num_attention_heads": heads,
        "num_layers": double,
        "num_single_layers": single,
        "out_channels": None,
        "patch_size": 1,
        "rope_theta": 2000,
        "timestep_guidance_channels": 256,
    }
    quantization = {
        "method": "svdquant",
        "rank": 32,
        "weight": {"dtype": "int4", "group_size": 64, "scale_dtype": None},
        "activation": {"dtype": "int4", "group_size": 64, "scale_dtype": None},
    }
    return {
        "model_class": "NunchakuFlux2Transformer2DModel",
        "config": json.dumps(config),
        "quantization_config": json.dumps(quantization),
    }


class KleinLoaderProfileTests(unittest.TestCase):
    def test_profiles_are_immutable(self):
        self.assertEqual([profile.name for profile in KLEIN_ARCHITECTURE_PROFILES], ["4B", "9B"])
        with self.assertRaises(FrozenInstanceError):
            KLEIN_ARCHITECTURE_PROFILES[0].context_dim = 1

    def test_resolves_exact_4b_profile(self):
        profile, config = _validated_comfy_config(
            metadata(context=7680, heads=24, double=5, single=20)
        )
        self.assertEqual(profile.name, "4B")
        self.assertEqual(config["hidden_size"], 3072)
        self.assertEqual(config["context_in_dim"], 7680)
        self.assertEqual(config["depth"], 5)
        self.assertEqual(config["depth_single_blocks"], 20)

    def test_resolves_exact_9b_profile(self):
        profile, config = _validated_comfy_config(
            metadata(context=12288, heads=32, double=8, single=24)
        )
        self.assertEqual(profile.name, "9B")
        self.assertEqual(config["hidden_size"], 4096)
        self.assertEqual(config["context_in_dim"], 12288)
        self.assertEqual(config["depth"], 8)
        self.assertEqual(config["depth_single_blocks"], 24)

    def test_rejects_mixed_profile(self):
        with self.assertRaisesRegex(ValueError, "architecture profile"):
            _validated_comfy_config(
                metadata(context=7680, heads=24, double=8, single=20)
            )

    def test_rejects_changed_shared_invariant(self):
        value = metadata(context=7680, heads=24, double=5, single=20)
        config = json.loads(value["config"])
        config["patch_size"] = 2
        value["config"] = json.dumps(config)
        with self.assertRaisesRegex(ValueError, '"patch_size"'):
            _validated_comfy_config(value)

    def test_rejects_changed_quantization_group_size(self):
        value = metadata(context=7680, heads=24, double=5, single=20)
        quantization = json.loads(value["quantization_config"])
        quantization["activation"]["group_size"] = 128
        value["quantization_config"] = json.dumps(quantization)
        with self.assertRaisesRegex(
            ValueError, "quantization_config.activation.group_size"
        ):
            _validated_comfy_config(value)

    def test_lora_metadata_is_profile_aware(self):
        path = pathlib.Path("example.safetensors")
        _validate_lora_profile(
            {"ss_base_model_version": "flux2_klein_4b"}, "4B", path
        )
        _validate_lora_profile(None, "4B", path)
        with self.assertRaisesRegex(ValueError, "not FLUX.2 Klein 9B"):
            _validate_lora_profile(
                {"ss_base_model_version": "flux2_klein_4b"}, "9B", path
            )

    def test_lora_cache_identity_includes_profile(self):
        path = pathlib.Path("example.safetensors")
        stat = SimpleNamespace(st_size=10, st_mtime_ns=20)
        self.assertNotEqual(
            _lora_cache_identity(path, stat, "4B"),
            _lora_cache_identity(path, stat, "9B"),
        )


if __name__ == "__main__":
    unittest.main()
