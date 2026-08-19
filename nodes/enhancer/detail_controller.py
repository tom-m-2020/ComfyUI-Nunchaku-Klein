"""Nunchaku-compatible FLUX.2 Klein detail controller.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

import logging

import torch

import comfy.model_management as model_management

from .common import TEXT_CATEGORY
from .conditioning import clone_conditioning, restore_dtype, temporary_float32
from .metadata import get_klein_sections
from .validation import validate_bool, validate_finite_range, validate_int_range


logger = logging.getLogger(__name__)


def _available_devices():
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


def _resolve_device(name):
    if name not in _available_devices():
        raise ValueError(
            f"device must be one of {_available_devices()}, got {name!r}."
        )
    if name == "auto":
        return model_management.get_torch_device()
    return torch.device(name)


class NunchakuKleinDetailController:
    """Scale exact encoder-provided section ranges or the legacy fallback."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",)},
            "optional": {
                "front_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Multiplier for the FRONT section. With Sectioned Encoder upstream this is the actual front token range; otherwise it's the first 25% of active tokens (arbitrary).",
                    },
                ),
                "mid_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Multiplier for the MID section.",
                    },
                ),
                "end_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Multiplier for the END section.",
                    },
                ),
                "emphasis_start": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Custom emphasis range start (token index).",
                    },
                ),
                "emphasis_end": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Custom emphasis range end (0 = disabled).",
                    },
                ),
                "emphasis_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Multiplier applied inside [emphasis_start, emphasis_end).",
                    },
                ),
                "preserve_original": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Linear blend back the unmodified active region. 0 = full effect.",
                    },
                ),
                "device": (_available_devices(), {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = TEXT_CATEGORY

    def control(
        self,
        conditioning,
        front_mult=1.0,
        mid_mult=1.0,
        end_mult=1.0,
        emphasis_start=0,
        emphasis_end=0,
        emphasis_mult=1.0,
        preserve_original=0.0,
        device="auto",
        debug=False,
    ):
        front_mult = validate_finite_range(
            front_mult, name="front_mult", minimum=0.0, maximum=10.0
        )
        mid_mult = validate_finite_range(
            mid_mult, name="mid_mult", minimum=0.0, maximum=10.0
        )
        end_mult = validate_finite_range(
            end_mult, name="end_mult", minimum=0.0, maximum=10.0
        )
        emphasis_start = validate_int_range(
            emphasis_start, name="emphasis_start", minimum=0, maximum=512
        )
        emphasis_end = validate_int_range(
            emphasis_end, name="emphasis_end", minimum=0, maximum=512
        )
        emphasis_mult = validate_finite_range(
            emphasis_mult, name="emphasis_mult", minimum=0.0, maximum=10.0
        )
        preserve_original = validate_finite_range(
            preserve_original,
            name="preserve_original",
            minimum=0.0,
            maximum=1.0,
        )
        debug = validate_bool(debug, name="debug")

        neutral = (
            front_mult == 1.0
            and mid_mult == 1.0
            and end_mult == 1.0
            and (emphasis_end == 0 or emphasis_mult == 1.0)
            and preserve_original == 0.0
        )
        if neutral or (
            isinstance(conditioning, (list, tuple)) and not conditioning
        ):
            return (conditioning,)

        compute_device = _resolve_device(device)
        output = clone_conditioning(conditioning)
        for index, item in enumerate(output):
            source = item[0]
            if source.ndim != 3:
                raise ValueError(
                    f"conditioning item {index} must have rank 3, got shape {list(source.shape)}."
                )

            working, original_dtype = temporary_float32(
                source.to(compute_device), clone=True
            )
            sequence_length = working.shape[1]
            active_end = sequence_length
            attention_mask = item[1].get("attention_mask")
            if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
                nonzero = attention_mask[0].nonzero()
                if len(nonzero) > 0:
                    active_end = min(int(nonzero[-1].item()) + 1, sequence_length)

            sections = get_klein_sections(item[1])
            if sections is None:
                front_end = int(active_end * 0.25)
                mid_end = int(active_end * 0.75)
                ranges = (
                    (0, front_end, front_mult),
                    (front_end, mid_end, mid_mult),
                    (mid_end, active_end, end_mult),
                )
                range_source = "fixed 25/50/25 fallback"
            else:
                ranges = (
                    (sections.front.start, sections.front.end, front_mult),
                    (sections.mid.start, sections.mid.end, mid_mult),
                    (sections.end.start, sections.end.end, end_mult),
                )
                range_source = "klein_sections"

            active = working[:, :active_end, :].clone()
            original_active = active.clone()

            def scale_range(start, end, multiplier):
                start = max(0, min(start, active_end))
                end = max(start, min(end, active_end))
                if multiplier != 1.0 and end > start:
                    active[:, start:end, :] *= multiplier

            for start, end, multiplier in ranges:
                scale_range(start, end, multiplier)
            if emphasis_end > 0 and emphasis_mult != 1.0:
                scale_range(emphasis_start, emphasis_end, emphasis_mult)
            if preserve_original > 0.0:
                active = (
                    active * (1.0 - preserve_original)
                    + original_active * preserve_original
                )

            result = source.detach().to("cpu").clone()
            result[:, :active_end, :] = restore_dtype(active, original_dtype).to("cpu")
            item[0] = result
            if debug:
                logger.info(
                    "Klein Detail Controller item %d: active=[0:%d), source=%s",
                    index,
                    active_end,
                    range_source,
                )

        return (output,)


__all__ = ["NunchakuKleinDetailController"]
