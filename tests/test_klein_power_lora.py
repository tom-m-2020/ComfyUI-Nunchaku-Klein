import importlib.util
import gc
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import weakref

import torch


TARGET = Path(__file__).resolve().parents[1]
COMFY = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_power_lora_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

LORA = sys.modules[f"{PACKAGE.__name__}.nodes.klein_lora"]
WRAPPER = sys.modules[f"{PACKAGE.__name__}.models.klein_wrapper"]


class FakeSharedState:
    def __init__(self):
        self.registered = []

    def register_patcher(self, patcher):
        self.registered.append(patcher)


class FakeAdapter:
    def __init__(self, profile="9B"):
        self.architecture_profile = profile
        self.shared_lora_state = FakeSharedState()


class FakeModel:
    def __init__(self, inherited=()):
        self.model = object()
        self.model_options = {
            "transformer_options": {WRAPPER.LORA_SPEC_OPTION: inherited}
        }
        self.clone_count = 0

    def clone(self):
        self.clone_count += 1
        clone = FakeModel()
        clone.model = self.model
        clone.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        return clone


def make_spec(name, strength):
    return WRAPPER.KleinLoraSpec(
        path=Path(name),
        size=len(name),
        mtime_ns=1,
        strength=float(strength),
        state_dict={"name": torch.tensor(len(name))},
    )


class KleinPowerLoraTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.adapter_patch = mock.patch.object(
            LORA, "_get_lora_adapter", return_value=self.adapter
        )
        self.adapter_patch.start()
        self.addCleanup(self.adapter_patch.stop)

    def fake_load_spec(self, _adapter, name, strength):
        if strength == 0.0:
            return None
        return make_spec(name, strength)

    def desired(self, model):
        return model.model_options["transformer_options"][WRAPPER.LORA_SPEC_OPTION]

    def test_power_ab_matches_two_ordinary_nodes(self):
        with mock.patch.object(
            LORA._KleinLoraSpecLoader, "load_spec", self.fake_load_spec
        ):
            ordinary_a = LORA.NunchakuKleinLoraLoader().load_lora(
                FakeModel(), "A.safetensors", 0.75
            )[0]
            ordinary_ab = LORA.NunchakuKleinLoraLoader().load_lora(
                ordinary_a, "B.safetensors", 1.25
            )[0]
            power_ab = LORA.NunchakuKleinPowerLoraLoader().load_loras(
                FakeModel(),
                json.dumps([
                    {"lora_name": "A.safetensors", "strength": 0.75},
                    {"lora_name": "B.safetensors", "strength": 1.25},
                ]),
            )[0]

        self.assertEqual(
            [entry.identity for entry in self.desired(ordinary_ab)],
            [entry.identity for entry in self.desired(power_ab)],
        )
        self.assertIsInstance(self.desired(power_ab), tuple)

    def test_inherited_a_then_power_bc_preserves_flat_order(self):
        inherited = (make_spec("A.safetensors", 0.5),)
        with mock.patch.object(
            LORA._KleinLoraSpecLoader, "load_spec", self.fake_load_spec
        ):
            branch = LORA.NunchakuKleinPowerLoraLoader().load_loras(
                FakeModel(inherited),
                json.dumps([
                    {"lora_name": "B.safetensors", "strength": 1.0},
                    {"lora_name": "C.safetensors", "strength": -0.25},
                ]),
            )[0]

        self.assertEqual(
            [(entry.path.name, entry.strength) for entry in self.desired(branch)],
            [("A.safetensors", 0.5), ("B.safetensors", 1.0), ("C.safetensors", -0.25)],
        )

    def test_disabled_zero_signed_and_row_order(self):
        seen = []

        def load_spec(_loader, _adapter, name, strength):
            seen.append((name, strength))
            return None if strength == 0.0 else make_spec(name, strength)

        with mock.patch.object(LORA._KleinLoraSpecLoader, "load_spec", load_spec):
            branch = LORA.NunchakuKleinPowerLoraLoader().load_loras(
                FakeModel(),
                json.dumps([
                    {"lora_name": "disabled", "strength": 9, "enabled": False},
                    {"lora_name": "zero", "strength": 0, "enabled": True},
                    {"lora_name": "negative", "strength": -1.5, "enabled": True},
                    {"lora_name": "positive", "strength": 2.0, "enabled": True},
                ]),
            )[0]

        self.assertEqual(seen, [("zero", 0.0), ("negative", -1.5), ("positive", 2.0)])
        self.assertEqual(
            [(entry.path.name, entry.strength) for entry in self.desired(branch)],
            [("negative", -1.5), ("positive", 2.0)],
        )

    def test_4b_and_9b_profile_metadata_validation(self):
        for profile in ("4B", "9B"):
            path = Path(f"klein-{profile}.safetensors")
            LORA._validate_lora_profile(
                {"ss_base_model_version": f"FLUX2-Klein-{profile}"}, profile, path
            )
            other = "9B" if profile == "4B" else "4B"
            with self.assertRaisesRegex(ValueError, f"Klein {profile}"):
                LORA._validate_lora_profile(
                    {"ss_base_model_version": f"FLUX2-Klein-{other}"}, profile, path
                )

    def test_invalid_row_fails_before_branch_is_cloned_or_registered(self):
        base = FakeModel()

        def load_spec(_loader, _adapter, name, strength):
            if name == "bad":
                raise ValueError("incompatible profile")
            return make_spec(name, strength)

        with mock.patch.object(LORA._KleinLoraSpecLoader, "load_spec", load_spec):
            with self.assertRaisesRegex(ValueError, "incompatible profile"):
                LORA.NunchakuKleinPowerLoraLoader().load_loras(
                    base,
                    json.dumps([
                        {"lora_name": "good", "strength": 1},
                        {"lora_name": "bad", "strength": 1},
                    ]),
                )

        self.assertEqual(base.clone_count, 0)
        self.assertEqual(self.adapter.shared_lora_state.registered, [])
        self.assertEqual(self.desired(base), ())

    def test_registration_keeps_the_ordinary_loader(self):
        self.assertIs(
            PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinLoraLoader"],
            LORA.NunchakuKleinLoraLoader,
        )
        self.assertIs(
            PACKAGE.NODE_CLASS_MAPPINGS["NunchakuKleinPowerLoraLoader"],
            LORA.NunchakuKleinPowerLoraLoader,
        )
        self.assertEqual(PACKAGE.WEB_DIRECTORY, "./web")

    def test_power_rows_schema_uses_one_frontend_custom_widget(self):
        required = LORA.NunchakuKleinPowerLoraLoader.INPUT_TYPES()["required"]

        self.assertEqual(set(required), {"model", "rows"})
        self.assertEqual(required["rows"][0], "STRING")
        self.assertTrue(required["rows"][1]["power_lora_rows"])
        self.assertIn("lora_names", required["rows"][1])

        frontend = (TARGET / "web" / "klein_power_lora_nodes2.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("getCustomWidgets()", frontend)
        self.assertIn('rows[0] = "KLEIN_POWER_LORA_ROWS"', frontend)
        self.assertIn('search.placeholder = "Search LoRAs"', frontend)
        self.assertIn("name.toLowerCase().includes(query)", frontend)
        self.assertIn("const openLoraPicker = ({ anchor, currentValue, onSelect })", frontend)
        self.assertIn('const option = document.createElement("div")', frontend)
        self.assertIn('option.setAttribute("role", "option")', frontend)
        self.assertIn("addRow(name);", frontend)
        self.assertNotIn("loraNames[0]", frontend)
        self.assertNotIn('document.createElement("select")', frontend)
        self.assertNotIn("computeSize =", frontend)
        self.assertNotIn("_lora_name", frontend)

    def test_parsed_lora_lru_evicts_and_releases_old_cache_entry(self):
        loader = LORA._KleinLoraSpecLoader(cache_entries=2)
        first_tensor = torch.ones(1)
        first_reference = weakref.ref(first_tensor)
        loader._cache_lora("first", {"weight": first_tensor}, {})
        loader._cache_lora("second", {"weight": torch.ones(1)}, {})
        del first_tensor

        self.assertIsNotNone(first_reference())
        loader._cache_lora("third", {"weight": torch.ones(1)}, {})
        gc.collect()

        self.assertEqual(list(loader.loaded_loras), ["second", "third"])
        self.assertIsNone(first_reference())

    def test_cache_hit_refreshes_lru_order_and_node_capacities_are_bounded(self):
        loader = LORA._KleinLoraSpecLoader(cache_entries=2)
        loader._cache_lora("first", {}, {})
        loader._cache_lora("second", {}, {})
        self.assertIsNotNone(loader._get_cached_lora("first"))
        loader._cache_lora("third", {}, {})

        self.assertEqual(list(loader.loaded_loras), ["first", "third"])
        self.assertEqual(LORA.NunchakuKleinLoraLoader().spec_loader.cache_entries, 1)
        self.assertEqual(
            LORA.NunchakuKleinPowerLoraLoader().spec_loader.cache_entries, 4
        )


if __name__ == "__main__":
    unittest.main()
