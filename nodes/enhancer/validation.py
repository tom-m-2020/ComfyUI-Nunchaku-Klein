"""Reusable validation at ComfyUI conditioning and latent boundaries."""

import math
from collections.abc import Sequence

import torch


def validate_conditioning(conditioning):
    """Validate the outer ComfyUI conditioning structure without copying it."""
    if conditioning is None:
        raise TypeError("conditioning must be a sequence, not None.")
    if not isinstance(conditioning, Sequence) or isinstance(
        conditioning, (str, bytes)
    ):
        raise TypeError("conditioning must be a sequence of tensor/metadata pairs.")
    for index, item in enumerate(conditioning):
        if not isinstance(item, Sequence) or len(item) < 2:
            raise TypeError(
                f"conditioning item {index} must contain a tensor and metadata."
            )
        if not torch.is_tensor(item[0]):
            raise TypeError(f"conditioning item {index} does not contain a tensor.")
        if not isinstance(item[1], dict):
            raise TypeError(f"conditioning item {index} metadata must be a dict.")
    return conditioning


def validate_bchw_tensor(tensor, *, name):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be BCHW, got shape {list(tensor.shape)}.")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must have positive dimensions.")
    return tensor


def validate_mask(mask):
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim not in (2, 3, 4):
        raise ValueError(
            f"mask must have rank 2, 3, or 4, got shape {list(mask.shape)}."
        )
    if mask.numel() == 0 or not torch.isfinite(mask).all().item():
        raise ValueError("mask must be non-empty and contain only finite values.")
    minimum = float(mask.min())
    maximum = float(mask.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"mask values must be within [0, 1], got [{minimum}, {maximum}]."
        )
    return mask


def validate_finite_range(value, *, name, minimum, maximum):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be within [{minimum}, {maximum}], got {value}."
        )
    return value


def validate_int_range(value, *, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name} must be within [{minimum}, {maximum}], got {value}."
        )
    return value


def validate_bool(value, *, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


__all__ = [
    "validate_bchw_tensor",
    "validate_bool",
    "validate_conditioning",
    "validate_finite_range",
    "validate_int_range",
    "validate_mask",
]
