"""Pure helpers for copying and temporarily transforming conditioning tensors."""

import torch

from .validation import validate_conditioning


def clone_conditioning(conditioning):
    """Shallow-copy conditioning entries and metadata, preserving tensors."""
    validate_conditioning(conditioning)
    return [[item[0], item[1].copy(), *item[2:]] for item in conditioning]


def temporary_float32(tensor, *, clone=True):
    """Return a float32 working tensor and the dtype needed for restoration."""
    if not torch.is_tensor(tensor):
        raise TypeError("temporary_float32 expects a torch.Tensor.")
    working = tensor.to(dtype=torch.float32)
    if clone:
        working = working.clone()
    return working, tensor.dtype


def restore_dtype(tensor, dtype):
    """Restore a temporary result to its boundary dtype."""
    if not torch.is_tensor(tensor):
        raise TypeError("restore_dtype expects a torch.Tensor.")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    return tensor.to(dtype=dtype)


__all__ = ["clone_conditioning", "restore_dtype", "temporary_float32"]
