"""Nunchaku-compatible FLUX.2 Klein text-conditioning enhancer.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

import logging
import math

import torch

from .common import TEXT_CATEGORY
from .conditioning import clone_conditioning, restore_dtype, temporary_float32
from .validation import validate_bool, validate_finite_range


logger = logging.getLogger(__name__)


class NunchakuKleinTextEnhancer:
    """Modify active text-conditioning tokens without patching the model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Scale text embeddings. <1=weaker prompt, >1=stronger",
                    },
                ),
            },
            "optional": {
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Token differentiation. >0=sharper, <0=blended",
                    },
                ),
                "normalize_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Equalize token magnitudes",
                    },
                ),
                "skip_bos": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Skip token 0 (BOS token with huge norm)",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = TEXT_CATEGORY
    DESCRIPTION = "Adjusts active FLUX.2 Klein text-conditioning embeddings."

    def enhance(
        self,
        conditioning,
        magnitude=1.0,
        contrast=0.0,
        normalize_strength=0.0,
        skip_bos=True,
        debug=False,
    ):
        magnitude = validate_finite_range(
            magnitude, name="magnitude", minimum=0.0, maximum=3.0
        )
        contrast = validate_finite_range(
            contrast, name="contrast", minimum=-1.0, maximum=2.0
        )
        normalize_strength = validate_finite_range(
            normalize_strength,
            name="normalize_strength",
            minimum=0.0,
            maximum=1.0,
        )
        skip_bos = validate_bool(skip_bos, name="skip_bos")
        debug = validate_bool(debug, name="debug")

        if not conditioning or (
            magnitude == 1.0
            and contrast == 0.0
            and normalize_strength == 0.0
        ):
            return (conditioning,)

        output = clone_conditioning(conditioning)
        for index, item in enumerate(output):
            source = item[0]
            if source.ndim != 3:
                raise ValueError(
                    "FLUX.2 Klein text conditioning must have shape "
                    f"[batch, tokens, embedding], got {list(source.shape)}."
                )

            working, original_dtype = temporary_float32(source)
            sequence_length = source.shape[1]
            attention_mask = item[1].get("attention_mask")
            if attention_mask is not None and attention_mask.dim() == 2:
                nonzero = attention_mask[0].nonzero()
                active_end = int(nonzero[-1].item()) + 1 if len(nonzero) else 77
            else:
                active_end = min(77, sequence_length)

            active_start = 1 if skip_bos else 0
            active = working[:, active_start:active_end, :]

            if debug:
                logger.info(
                    "Text Enhancer item %d: active tokens [%d:%d], initial mean norm %.6f",
                    index,
                    active_start,
                    active_end,
                    float(active.norm(dim=-1).mean()),
                )

            if normalize_strength > 0.0:
                norms = active.norm(dim=-1, keepdim=True)
                mean_norm = norms.mean()
                normalized = active / (norms + 1e-8) * mean_norm
                active = (
                    active * (1.0 - normalize_strength)
                    + normalized * normalize_strength
                )

            if contrast != 0.0:
                sequence_mean = active.mean(dim=1, keepdim=True)
                contrast_scale = 1.0 + contrast if contrast >= 0.0 else math.exp(contrast)
                active = sequence_mean + (active - sequence_mean) * contrast_scale

            if magnitude != 1.0:
                active = active * magnitude

            result = source.clone()
            result[:, active_start:active_end, :] = restore_dtype(
                active, original_dtype
            )
            item[0] = result

            if debug:
                logger.info(
                    "Text Enhancer item %d: final mean norm %.6f",
                    index,
                    float(active.norm(dim=-1).mean()),
                )

        return (output,)


__all__ = ["NunchakuKleinTextEnhancer"]
