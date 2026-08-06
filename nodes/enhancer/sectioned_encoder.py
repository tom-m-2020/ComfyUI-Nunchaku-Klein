"""Nunchaku-compatible FLUX.2 Klein sectioned text encoder.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

import logging
import re
from collections.abc import Mapping, Sequence

from .common import KLEIN_SECTIONS_KEY, TEXT_CATEGORY
from .validation import validate_bool


logger = logging.getLogger(__name__)

_KLEIN_CHAT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
_SEPARATORS = {
    "comma": ", ",
    "period": ". ",
    "space": " ",
    "newline": "\n",
}


def _parse_marker_sections(text):
    if not text:
        return None
    pattern = r"\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    sections = {"front": "", "mid": "", "end": ""}
    for name, content in matches:
        sections[name.lower()] = content.strip()
    return sections


def _tokenizer_contract(clip):
    tokenize = getattr(clip, "tokenize", None)
    encode = getattr(clip, "encode_from_tokens", None)
    if not callable(tokenize) or not callable(encode):
        raise TypeError(
            "clip must expose callable tokenize() and encode_from_tokens() methods."
        )

    outer = getattr(clip, "tokenizer", None)
    if outer is None:
        raise RuntimeError(
            "Klein Sectioned Encoder requires clip.tokenizer, but it is absent."
        )

    available = [
        name for name in ("qwen3_8b", "qwen3_4b") if getattr(outer, name, None)
    ]
    if len(available) != 1:
        raise RuntimeError(
            "Klein Sectioned Encoder requires exactly one of "
            "clip.tokenizer.qwen3_8b or clip.tokenizer.qwen3_4b; "
            f"found {available or 'neither'}."
        )

    wrapper = getattr(outer, "llama_template", None)
    if wrapper != _KLEIN_CHAT_TEMPLATE:
        raise RuntimeError(
            "Unsupported Klein chat wrapper at clip.tokenizer.llama_template. "
            "The current ComfyUI Klein wrapper contract has changed."
        )

    inner = getattr(outer, available[0])
    tokenizer = getattr(inner, "tokenizer", None)
    if not callable(tokenizer):
        raise RuntimeError(
            f"clip.tokenizer.{available[0]}.tokenizer must be callable."
        )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Klein Sectioned Encoder requires a fast Qwen tokenizer with "
            "offset mappings; this tokenizer does not expose that contract."
        )
    return tokenize, encode, tokenizer, wrapper


def _encode_with_offsets(tokenizer, text):
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise RuntimeError(
            "The Klein Qwen tokenizer could not return offset mappings; "
            "section ranges cannot be determined safely."
        ) from error
    if not isinstance(encoded, Mapping):
        raise TypeError("the Klein Qwen tokenizer result must be a mapping.")
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (
        not isinstance(input_ids, Sequence)
        or not isinstance(offsets, Sequence)
        or len(input_ids) != len(offsets)
    ):
        raise RuntimeError(
            "the Klein Qwen tokenizer returned inconsistent input_ids and "
            "offset_mapping sequences."
        )
    normalized = []
    for index, offset in enumerate(offsets):
        if (
            not isinstance(offset, Sequence)
            or len(offset) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
        ):
            raise RuntimeError(
                f"offset_mapping[{index}] must contain two integer character offsets."
            )
        start, end = offset
        if start < 0 or end < start:
            raise RuntimeError(f"offset_mapping[{index}] is not a valid range.")
        normalized.append((start, end))
    return normalized


def _compute_section_ranges(tokenizer, wrapper, sections, separator):
    prefix, suffix = wrapper.split("{}")
    del suffix

    spans = {}
    prompt_parts = []
    cursor = len(prefix)
    has_content = False
    for name in ("front", "mid", "end"):
        text = sections[name]
        if text:
            if has_content:
                prompt_parts.append(separator)
                cursor += len(separator)
            start = cursor
            prompt_parts.append(text)
            cursor += len(text)
            spans[name] = (start, cursor)
            has_content = True
        else:
            spans[name] = (cursor, cursor)

    full_prompt = "".join(prompt_parts)
    wrapped_prompt = wrapper.format(full_prompt)
    offsets = _encode_with_offsets(tokenizer, wrapped_prompt)

    def token_range(span):
        start, end = span
        if start == end:
            position = sum(1 for _, token_end in offsets if token_end <= start)
            return (position, position)
        indexes = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if not indexes:
            raise RuntimeError(
                f"the tokenizer produced no tokens for non-empty character range [{start}:{end})."
            )
        return (indexes[0], indexes[-1] + 1)

    return full_prompt, {name: token_range(spans[name]) for name in spans}


class NunchakuKleinSectionedEncoder:
    """Encode one Klein prompt and attach exact section token ranges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "front_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "FRONT section text.",
                    },
                ),
                "mid_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "MID section text.",
                    },
                ),
                "end_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "END section text.",
                    },
                ),
                "combined_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional. Single prompt with [FRONT]/[MID]/[END] markers — overrides the three text boxes when non-empty and contains markers.",
                    },
                ),
                "separator": (
                    list(_SEPARATORS),
                    {
                        "default": "comma",
                        "tooltip": "How to join sections in the final prompt sent to Klein.",
                    },
                ),
                "show_preview": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "conditioning",
        "front_section",
        "mid_section",
        "end_section",
        "full_prompt",
    )
    FUNCTION = "encode_sectioned"
    CATEGORY = TEXT_CATEGORY
    OUTPUT_NODE = True

    def encode_sectioned(
        self,
        clip,
        front_text="",
        mid_text="",
        end_text="",
        combined_prompt="",
        separator="comma",
        show_preview=True,
        debug=False,
    ):
        for name, value in (
            ("front_text", front_text),
            ("mid_text", mid_text),
            ("end_text", end_text),
            ("combined_prompt", combined_prompt),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")
        if separator not in _SEPARATORS:
            raise ValueError(
                f"separator must be one of {list(_SEPARATORS)}, got {separator!r}."
            )
        show_preview = validate_bool(show_preview, name="show_preview")
        debug = validate_bool(debug, name="debug")

        marker_sections = _parse_marker_sections(combined_prompt)
        sections = marker_sections or {
            "front": front_text or "",
            "mid": mid_text or "",
            "end": end_text or "",
        }
        separator_text = _SEPARATORS[separator]
        tokenize, encode, tokenizer, wrapper = _tokenizer_contract(clip)
        full_prompt, ranges = _compute_section_ranges(
            tokenizer, wrapper, sections, separator_text
        )

        tokens = tokenize(full_prompt)
        try:
            encoded = encode(tokens, return_dict=True)
        except TypeError as error:
            raise RuntimeError(
                "clip.encode_from_tokens() must support return_dict=True so "
                "existing conditioning metadata can be preserved."
            ) from error
        if not isinstance(encoded, Mapping) or "cond" not in encoded:
            raise RuntimeError(
                "clip.encode_from_tokens(..., return_dict=True) must return a "
                "mapping containing 'cond'."
            )
        metadata = dict(encoded)
        conditioning_tensor = metadata.pop("cond")
        metadata[KLEIN_SECTIONS_KEY] = ranges

        if show_preview or debug:
            self._print_preview(sections, full_prompt, ranges, separator, tokenizer)

        return (
            [[conditioning_tensor, metadata]],
            sections["front"],
            sections["mid"],
            sections["end"],
            full_prompt,
        )

    @staticmethod
    def _print_preview(sections, full_prompt, ranges, separator, tokenizer):
        count = lambda text: len(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        prefix, suffix = _KLEIN_CHAT_TEMPLATE.split("{}")
        lines = ["", "=" * 70, "FLUX.2 Klein Sectioned Encoding (v2)", "=" * 70]
        lines.append(f"Separator: {separator!r}")
        lines.append("Section token counts (HF-tokenizer-exact, no padding):")
        for name in ("front", "mid", "end"):
            lines.append(
                f"  {name.upper():5s}: {count(sections[name])} tokens   {sections[name]!r}"
            )
        lines.append(
            f"Klein wrapper overhead: prefix={count(prefix)} suffix={count(suffix)} tokens"
        )
        lines.append("Encoded-sequence section ranges (used by Detail Controller):")
        for name in ("front", "mid", "end"):
            start, end = ranges[name]
            lines.append(
                f"  {name.upper():5s}  tokens [{start}:{end})  span={end - start}"
            )
        lines.append(
            "Wire this conditioning into Detail Controller — it will scale these exact ranges."
        )
        lines.append("-" * 70)
        lines.append(f"Final prompt: {full_prompt!r}")
        lines.append("=" * 70)
        logger.info("\n".join(lines))


__all__ = ["NunchakuKleinSectionedEncoder"]
