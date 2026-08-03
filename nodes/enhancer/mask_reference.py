"""Target-owned spatial attenuation for encoded FLUX.2 references.

Behavior is adapted from ComfyUI-Flux2Klein-Enhancer under the MIT License;
see THIRD_PARTY_NOTICES.md.
"""

import logging

import torch
import torch.nn.functional as functional

from .common import REFERENCE_CATEGORY, REFERENCE_LATENTS_KEY
from .conditioning import clone_conditioning, restore_dtype, temporary_float32
from .metadata import get_reference_latents
from .validation import (
    validate_finite_range,
    validate_int_range,
    validate_mask,
)


logger = logging.getLogger(__name__)


def _resize_mask(mask, height, width):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask[:1].unsqueeze(1)
    else:
        mask = mask[:1, :1]
    return functional.interpolate(
        mask.float(), size=(height, width), mode="bilinear", align_corners=False
    )


def _feather_mask(mask, radius):
    if radius == 0:
        return mask
    kernel_size = radius * 2 + 1
    sigma = max(radius / 3.0, 1e-6)
    axis = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
    axis = axis - radius
    gaussian = torch.exp(-0.5 * (axis / sigma) ** 2)
    gaussian = gaussian / gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return functional.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)


class NunchakuKleinMaskReferenceController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "step": 1},
                ),
                "reference_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 7, "step": 1},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = "Attenuates one encoded reference latent with a spatial mask."

    def apply_mask(
        self,
        conditioning,
        mask,
        strength=1.0,
        invert_mask=False,
        feather=0,
        reference_index=0,
        debug=False,
    ):
        strength = validate_finite_range(
            strength, name="strength", minimum=0.0, maximum=1.0
        )
        feather = validate_int_range(
            feather, name="feather", minimum=0, maximum=64
        )
        reference_index = validate_int_range(
            reference_index, name="reference_index", minimum=0, maximum=7
        )
        validate_mask(mask)

        if strength == 0.0:
            if debug:
                logger.info("Mask Ref Controller: strength is zero; no change.")
            return (conditioning,)

        output = clone_conditioning(conditioning)
        for item in output:
            references = get_reference_latents(item[1])
            if reference_index >= len(references):
                if debug:
                    logger.info(
                        "Mask Ref Controller: no reference at index %d.",
                        reference_index,
                    )
                continue

            reference, original_dtype = temporary_float32(
                references[reference_index]
            )
            spatial_mask = _resize_mask(mask, reference.shape[-2], reference.shape[-1])
            if invert_mask:
                spatial_mask = 1.0 - spatial_mask
            spatial_mask = _feather_mask(spatial_mask, feather).to(reference.device)
            multiplier = 1.0 - strength * (1.0 - spatial_mask)
            modified = restore_dtype(reference * multiplier, original_dtype)

            if debug:
                logger.info(
                    "Mask Ref Controller: reference=%d shape=%s strength=%.3f "
                    "feather=%d invert=%s mean_attenuation=%.3f",
                    reference_index,
                    list(reference.shape),
                    strength,
                    feather,
                    invert_mask,
                    float(1.0 - multiplier.mean()),
                )

            updated = list(references)
            updated[reference_index] = modified
            item[1][REFERENCE_LATENTS_KEY] = updated

        return (output,)


__all__ = ["NunchakuKleinMaskReferenceController"]
