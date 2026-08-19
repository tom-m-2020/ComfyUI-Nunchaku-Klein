import importlib.util
import pathlib
import sys
import unittest

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))

spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_test_package",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from comfy.text_encoders.flux import KleinTokenizer, KleinTokenizer8B

from nunchaku_klein_test_package.nodes import enhancer as ENHANCER
from nunchaku_klein_test_package.nodes.enhancer.common import (
    KLEIN_SECTIONS_KEY,
    TEXT_CATEGORY,
)


class FakeClip:
    def __init__(self, profile="9B", *, batch=1, width=1):
        self.profile = profile
        self.tokenizer = KleinTokenizer() if profile == "4B" else KleinTokenizer8B()
        self.batch = batch
        self.width = width
        self.last_prompt = None
        self.last_tokens = None
        self.last_encoded_cond = None

    def tokenize(self, prompt):
        self.last_prompt = prompt
        self.last_tokens = self.tokenizer.tokenize_with_weights(prompt)
        return self.last_tokens

    def encode_from_tokens(self, tokens, return_dict=False):
        if not return_dict:
            raise AssertionError("Sectioned Encoder must preserve the metadata dict.")
        key = "qwen3_4b" if self.profile == "4B" else "qwen3_8b"
        sequence_length = len(tokens[key][0])
        self.last_encoded_cond = torch.arange(
            self.batch * sequence_length * self.width, dtype=torch.float16
        ).reshape(self.batch, sequence_length, self.width)
        return {
            "cond": self.last_encoded_cond,
            "pooled_output": torch.tensor([3.0]),
            "attention_mask": torch.ones((1, sequence_length)),
            "existing": {"kept": True},
        }


class SectionedEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = FakeClip()

    def test_registration_category_and_exact_ui_contract(self):
        node_id = "NunchakuKleinSectionedEncoder"
        self.assertIs(
            PACKAGE.NODE_CLASS_MAPPINGS[node_id],
            ENHANCER.NunchakuKleinSectionedEncoder,
        )
        node = PACKAGE.NODE_CLASS_MAPPINGS[node_id]
        self.assertEqual(node.CATEGORY, TEXT_CATEGORY)
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[node_id],
            "Nunchaku FLUX.2 Klein Sectioned Encoder",
        )
        self.assertEqual(
            node.RETURN_TYPES,
            ("CONDITIONING", "STRING", "STRING", "STRING", "STRING"),
        )
        self.assertEqual(
            node.RETURN_NAMES,
            (
                "conditioning",
                "front_section",
                "mid_section",
                "end_section",
                "full_prompt",
            ),
        )
        self.assertTrue(node.OUTPUT_NODE)
        inputs = node.INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["clip"])
        self.assertEqual(
            list(inputs["optional"]),
            [
                "front_text",
                "mid_text",
                "end_text",
                "combined_prompt",
                "separator",
                "show_preview",
                "debug",
            ],
        )
        self.assertEqual(inputs["optional"]["separator"][0], ["comma", "period", "space", "newline"])
        self.assertEqual(inputs["optional"]["separator"][1]["default"], "comma")

    def encode(self, **kwargs):
        kwargs.setdefault("show_preview", False)
        return ENHANCER.NunchakuKleinSectionedEncoder().encode_sectioned(
            self.clip, **kwargs
        )

    def test_final_prompt_exact_ranges_and_metadata_preservation(self):
        conditioning, front, mid, end, prompt = self.encode(
            front_text="front", mid_text="middle", end_text="end"
        )
        self.assertEqual((front, mid, end), ("front", "middle", "end"))
        self.assertEqual(prompt, "front, middle, end")
        self.assertEqual(self.clip.last_prompt, prompt)
        metadata = conditioning[0][1]
        self.assertEqual(
            metadata[KLEIN_SECTIONS_KEY],
            {"front": (3, 4), "mid": (5, 6), "end": (7, 8)},
        )
        self.assertEqual(metadata["existing"], {"kept": True})
        self.assertIn("pooled_output", metadata)
        self.assertIn("attention_mask", metadata)

    def test_4b_and_9b_profiles_have_exact_same_token_section_contract(self):
        expected = {"front": (3, 4), "mid": (5, 6), "end": (7, 8)}
        for profile in ("4B", "9B"):
            with self.subTest(profile=profile):
                clip = FakeClip(profile, batch=2, width=3)
                result = ENHANCER.NunchakuKleinSectionedEncoder().encode_sectioned(
                    clip,
                    front_text="front",
                    mid_text="middle",
                    end_text="end",
                    show_preview=False,
                )
                tensor, metadata = result[0][0]
                self.assertEqual(metadata[KLEIN_SECTIONS_KEY], expected)
                self.assertIs(tensor, clip.last_encoded_cond)
                self.assertEqual(tensor.shape, (2, 512, 3))
                self.assertEqual(tensor.dtype, torch.float16)

    def test_each_section_and_its_boundaries_are_independent(self):
        cases = {
            "front": {"front": (3, 4), "mid": (4, 4), "end": (4, 4)},
            "mid": {"front": (3, 3), "mid": (3, 4), "end": (4, 4)},
            "end": {"front": (3, 3), "mid": (3, 3), "end": (3, 4)},
        }
        for name, expected in cases.items():
            with self.subTest(section=name):
                result = self.encode(**{f"{name}_text": name})
                ranges = result[0][0][1][KLEIN_SECTIONS_KEY]
                self.assertEqual(ranges, expected)
                start, end = ranges[name]
                self.assertEqual(end - start, 1)
                self.assertEqual(result[0][0][0][0, start].item(), float(start))

    def test_section_metadata_does_not_modify_native_conditioning_values(self):
        clip = FakeClip("4B", batch=2, width=3)
        node_result = ENHANCER.NunchakuKleinSectionedEncoder().encode_sectioned(
            clip,
            front_text="front",
            mid_text="middle",
            end_text="end",
            show_preview=False,
        )
        sectioned = node_result[0][0][0]
        direct = clip.encode_from_tokens(clip.tokenize(node_result[4]), return_dict=True)[
            "cond"
        ]
        self.assertTrue(torch.equal(sectioned, direct))
        for start, end in node_result[0][0][1][KLEIN_SECTIONS_KEY].values():
            self.assertEqual(
                torch.mean(torch.abs(sectioned[:, start:end] - direct[:, start:end])).item()
                if end > start
                else 0.0,
                0.0,
            )

    def test_empty_sections_and_combined_marker_override(self):
        result = self.encode(front_text="front", mid_text="", end_text="end")
        self.assertEqual(result[4], "front, end")
        ranges = result[0][0][1][KLEIN_SECTIONS_KEY]
        self.assertEqual(ranges["front"], (3, 4))
        self.assertEqual(ranges["mid"], (4, 4))
        self.assertEqual(ranges["end"], (5, 6))

        combined = self.encode(
            front_text="ignored",
            combined_prompt="[END] finish\nline [FRONT] 猫 café [MID] middle",
            separator="newline",
        )
        self.assertEqual(combined[1:4], ("猫 café", "middle", "finish\nline"))
        self.assertEqual(combined[4], "猫 café\nmiddle\nfinish\nline")
        sections = combined[0][0][1][KLEIN_SECTIONS_KEY]
        self.assertTrue(sections["front"][0] < sections["front"][1])
        self.assertTrue(sections["mid"][0] < sections["mid"][1])
        self.assertTrue(sections["end"][0] < sections["end"][1])

        empty = self.encode()
        self.assertEqual(empty[1:], ("", "", "", ""))
        self.assertEqual(
            empty[0][0][1][KLEIN_SECTIONS_KEY],
            {"front": (3, 3), "mid": (3, 3), "end": (3, 3)},
        )

    def test_unicode_multiline_and_near_limit_are_not_truncated(self):
        unicode_result = self.encode(
            front_text="猫 café", mid_text="line one\nline two", end_text="終わり",
            separator="newline",
        )
        ranges = unicode_result[0][0][1][KLEIN_SECTIONS_KEY]
        self.assertLessEqual(ranges["front"][1], ranges["mid"][0])
        self.assertLessEqual(ranges["mid"][1], ranges["end"][0])

        long_text = " ".join(["token"] * 600)
        long_result = self.encode(front_text=long_text)
        tensor = long_result[0][0][0]
        long_range = long_result[0][0][1][KLEIN_SECTIONS_KEY]["front"]
        self.assertGreater(tensor.shape[1], 512)
        self.assertEqual(long_range, (3, 603))
        self.assertLessEqual(long_range[1], tensor.shape[1])

    def test_capability_failures_are_actionable(self):
        node = ENHANCER.NunchakuKleinSectionedEncoder()
        with self.assertRaisesRegex(TypeError, "tokenize.*encode_from_tokens"):
            node.encode_sectioned(object(), show_preview=False)

        class MissingQwen:
            tokenize = lambda self, text: text
            encode_from_tokens = lambda self, tokens, return_dict=False: {}
            tokenizer = object()

        with self.assertRaisesRegex(RuntimeError, "qwen3_8b"):
            node.encode_sectioned(MissingQwen(), show_preview=False)

        ambiguous = FakeClip("4B")
        ambiguous.tokenizer.qwen3_8b = ambiguous.tokenizer.qwen3_4b
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            node.encode_sectioned(ambiguous, show_preview=False)

        original = self.clip.tokenizer.llama_template
        try:
            self.clip.tokenizer.llama_template = "{}"
            with self.assertRaisesRegex(RuntimeError, "chat wrapper"):
                node.encode_sectioned(self.clip, show_preview=False)
        finally:
            self.clip.tokenizer.llama_template = original


class DetailControllerTests(unittest.TestCase):
    def setUp(self):
        self.node = ENHANCER.NunchakuKleinDetailController()

    def conditioning(
        self, *, dtype=torch.float16, metadata=None, batch=2, length=12, width=3
    ):
        tensor = torch.ones((batch, length, width), dtype=dtype)
        return [[tensor, dict(metadata or {})]], tensor

    def test_registration_category_and_exact_ui_contract(self):
        node_id = "NunchakuKleinDetailController"
        self.assertIs(PACKAGE.NODE_CLASS_MAPPINGS[node_id], type(self.node))
        self.assertEqual(type(self.node).CATEGORY, TEXT_CATEGORY)
        self.assertEqual(
            PACKAGE.NODE_DISPLAY_NAME_MAPPINGS[node_id],
            "Nunchaku FLUX.2 Klein Detail Controller",
        )
        inputs = type(self.node).INPUT_TYPES()
        self.assertEqual(list(inputs["required"]), ["conditioning"])
        self.assertEqual(
            list(inputs["optional"]),
            [
                "front_mult",
                "mid_mult",
                "end_mult",
                "emphasis_start",
                "emphasis_end",
                "emphasis_mult",
                "preserve_original",
                "device",
                "debug",
            ],
        )
        self.assertEqual(inputs["optional"]["front_mult"][1]["step"], 0.05)
        self.assertEqual(inputs["optional"]["emphasis_mult"][1]["step"], 0.1)

    def test_neutral_is_exact_object_passthrough(self):
        conditioning, tensor = self.conditioning()
        output = self.node.control(conditioning)[0]
        self.assertIs(output, conditioning)
        self.assertIs(output[0][0], tensor)
        self.assertIs(output[0][1], conditioning[0][1])

    def test_sectioned_encoder_composition_is_exact_for_4b_and_9b(self):
        for profile in ("4B", "9B"):
            with self.subTest(profile=profile):
                clip = FakeClip(profile, batch=2, width=3)
                sectioned = ENHANCER.NunchakuKleinSectionedEncoder().encode_sectioned(
                    clip,
                    front_text="front",
                    mid_text="middle",
                    end_text="end",
                    show_preview=False,
                )[0]
                source = sectioned[0][0]
                snapshot = source.clone()
                metadata = sectioned[0][1]
                output = self.node.control(
                    sectioned,
                    front_mult=2.0,
                    mid_mult=3.0,
                    end_mult=4.0,
                    device="cpu",
                )[0]
                result = output[0][0]
                self.assertTrue(torch.equal(result[:, :3], snapshot[:, :3]))
                self.assertTrue(torch.equal(result[:, 3:4], snapshot[:, 3:4] * 2))
                self.assertTrue(torch.equal(result[:, 4:5], snapshot[:, 4:5]))
                self.assertTrue(torch.equal(result[:, 5:6], snapshot[:, 5:6] * 3))
                self.assertTrue(torch.equal(result[:, 6:7], snapshot[:, 6:7]))
                self.assertTrue(torch.equal(result[:, 7:8], snapshot[:, 7:8] * 4))
                self.assertTrue(torch.equal(result[:, 8:], snapshot[:, 8:]))
                self.assertTrue(torch.equal(source, snapshot))
                self.assertEqual(output[0][1], metadata)
                self.assertIsNot(output[0][1], metadata)

    def test_context_width_is_not_a_controller_mapping(self):
        sections = {
            KLEIN_SECTIONS_KEY: {
                "front": (1, 2),
                "mid": (3, 4),
                "end": (5, 6),
            }
        }
        for profile, width in (("4B", 7680), ("9B", 12288)):
            with self.subTest(profile=profile):
                conditioning, source = self.conditioning(
                    batch=1, length=8, width=width, metadata=sections
                )
                result = self.node.control(
                    conditioning,
                    front_mult=2.0,
                    mid_mult=3.0,
                    end_mult=4.0,
                    device="cpu",
                )[0][0][0]
                self.assertEqual(result.shape, (1, 8, width))
                self.assertTrue(torch.equal(result[:, 1], source[:, 1] * 2))
                self.assertTrue(torch.equal(result[:, 3], source[:, 3] * 3))
                self.assertTrue(torch.equal(result[:, 5], source[:, 5] * 4))
                self.assertTrue(torch.equal(result[:, 0], source[:, 0]))
                self.assertTrue(torch.equal(result[:, 6:], source[:, 6:]))

    def test_exact_metadata_ranges_overlap_order_and_clamping(self):
        metadata = {
            KLEIN_SECTIONS_KEY: {
                "front": (1, 5),
                "mid": (3, 8),
                "end": (8, 99),
            },
            "attention_mask": torch.tensor([[1] * 10 + [0, 0]]),
            "marker": object(),
        }
        conditioning, tensor = self.conditioning(metadata=metadata)
        snapshot = tensor.clone()
        output = self.node.control(
            conditioning, front_mult=2.0, mid_mult=3.0, end_mult=4.0, device="cpu"
        )[0]
        result = output[0][0]
        self.assertEqual(result.dtype, tensor.dtype)
        self.assertEqual(result.shape[0], 2)
        self.assertTrue(torch.equal(result[:, :1], snapshot[:, :1]))
        self.assertTrue(torch.equal(result[:, 1:3], snapshot[:, 1:3] * 2))
        self.assertTrue(torch.equal(result[:, 3:5], snapshot[:, 3:5] * 6))
        self.assertTrue(torch.equal(result[:, 5:8], snapshot[:, 5:8] * 3))
        self.assertTrue(torch.equal(result[:, 8:10], snapshot[:, 8:10] * 4))
        self.assertTrue(torch.equal(result[:, 10:], snapshot[:, 10:]))
        self.assertTrue(torch.equal(tensor, snapshot))
        self.assertIsNot(output[0][1], metadata)
        self.assertIs(output[0][1]["marker"], metadata["marker"])

    def test_legacy_fallback_is_exact_25_50_25(self):
        conditioning, tensor = self.conditioning(length=12, dtype=torch.float32)
        result = self.node.control(
            conditioning, front_mult=2.0, mid_mult=3.0, end_mult=4.0, device="cpu"
        )[0][0][0]
        self.assertTrue(torch.equal(result[:, :3], tensor[:, :3] * 2))
        self.assertTrue(torch.equal(result[:, 3:9], tensor[:, 3:9] * 3))
        self.assertTrue(torch.equal(result[:, 9:], tensor[:, 9:] * 4))

    def test_emphasis_applies_after_sections_and_preserve_blends(self):
        metadata = {
            KLEIN_SECTIONS_KEY: {
                "front": (0, 4),
                "mid": (4, 8),
                "end": (8, 12),
            }
        }
        conditioning, tensor = self.conditioning(metadata=metadata, dtype=torch.float32)
        result = self.node.control(
            conditioning,
            front_mult=2.0,
            emphasis_start=2,
            emphasis_end=6,
            emphasis_mult=3.0,
            preserve_original=0.5,
            device="cpu",
        )[0][0][0]
        self.assertTrue(torch.equal(result[:, :2], tensor[:, :2] * 1.5))
        self.assertTrue(torch.equal(result[:, 2:4], tensor[:, 2:4] * 3.5))
        self.assertTrue(torch.equal(result[:, 4:6], tensor[:, 4:6] * 2.0))
        self.assertTrue(torch.equal(result[:, 6:], tensor[:, 6:]))

    def test_invalid_section_metadata_is_rejected(self):
        invalid = {KLEIN_SECTIONS_KEY: {"front": (0, 2), "mid": (2, 4)}}
        conditioning, _ = self.conditioning(metadata=invalid)
        with self.assertRaisesRegex(TypeError, r"klein_sections\['end'\]"):
            self.node.control(conditioning, front_mult=2.0, device="cpu")

        reversed_range = {
            KLEIN_SECTIONS_KEY: {
                "front": (3, 2),
                "mid": (2, 4),
                "end": (4, 5),
            }
        }
        conditioning, _ = self.conditioning(metadata=reversed_range)
        with self.assertRaisesRegex(ValueError, "valid half-open range"):
            self.node.control(conditioning, front_mult=2.0, device="cpu")

    def test_numeric_and_boolean_inputs_are_validated(self):
        conditioning, _ = self.conditioning()
        with self.assertRaisesRegex(ValueError, "front_mult"):
            self.node.control(conditioning, front_mult=float("nan"))
        with self.assertRaisesRegex(TypeError, "emphasis_end"):
            self.node.control(conditioning, emphasis_end=True)
        with self.assertRaisesRegex(TypeError, "debug"):
            self.node.control(conditioning, front_mult=2.0, debug=1)
        with self.assertRaisesRegex(TypeError, "conditioning"):
            self.node.control(None, front_mult=2.0, device="cpu")


if __name__ == "__main__":
    unittest.main()
