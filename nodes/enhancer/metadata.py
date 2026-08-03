"""Validated accessors for shared FLUX.2 Klein conditioning metadata."""
"""Reference latents are stored as detached shared views. Compatibility nodes must treat them as immutable and create a private clone before any in-place modification."""


from collections.abc import Mapping, Sequence

import torch

from .common import (
    KLEIN_SECTIONS_KEY,
    REFERENCE_LATENTS_KEY,
    REFERENCE_METHOD_KEY,
    KleinSections,
    TokenRange,
)
from .validation import validate_bchw_tensor


def latent_samples(value, *, name="latent"):
    """Extract and validate BCHW samples from a ComfyUI LATENT or tensor."""
    if isinstance(value, Mapping):
        if "samples" not in value:
            raise ValueError(f"{name} does not contain a 'samples' tensor.")
        value = value["samples"]
    return validate_bchw_tensor(value, name=name)


def split_latent_batch(value, *, name="latent"):
    """Split a LATENT batch into ordered, detached batch-one views."""
    samples = latent_samples(value, name=name)
    return tuple(samples[index : index + 1].detach() for index in range(samples.shape[0]))


def get_reference_latents(metadata, *, required=False):
    """Return a validated immutable view of ordered reference latents."""
    if not isinstance(metadata, Mapping):
        raise TypeError("conditioning metadata must be a mapping.")
    references = metadata.get(REFERENCE_LATENTS_KEY)
    if references is None:
        if required:
            raise ValueError("conditioning contains no reference_latents.")
        return ()
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise TypeError("reference_latents must be an ordered sequence of tensors.")
    validated = tuple(
        validate_bchw_tensor(reference, name=f"reference_latents[{index}]")
        for index, reference in enumerate(references)
    )
    if required and not validated:
        raise ValueError("conditioning reference_latents is empty.")
    return validated


def set_reference_latents(metadata, references, *, method="index"):
    """Copy metadata and set an ordered reference list and method."""
    if not isinstance(metadata, Mapping):
        raise TypeError("conditioning metadata must be a mapping.")
    references = tuple(
        validate_bchw_tensor(reference, name=f"reference_latents[{index}]")
        for index, reference in enumerate(references)
    )
    if not references:
        raise ValueError("at least one reference latent is required.")
    if method != "index":
        raise ValueError(
            f"only the validated FLUX.2 reference method 'index' is supported, got {method!r}."
        )
    result = dict(metadata)
    result[REFERENCE_LATENTS_KEY] = list(references)
    result[REFERENCE_METHOD_KEY] = method
    return result


def get_klein_sections(metadata):
    """Validate and return immutable section ranges, or ``None`` when absent."""
    if not isinstance(metadata, Mapping):
        raise TypeError("conditioning metadata must be a mapping.")
    raw = metadata.get(KLEIN_SECTIONS_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("klein_sections must be a mapping.")

    def token_range(name):
        value = raw.get(name)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise TypeError(f"klein_sections[{name!r}] must be two integers.")
        start, end = value
        if start < 0 or end < start:
            raise ValueError(f"klein_sections[{name!r}] is not a valid half-open range.")
        return TokenRange(start, end)

    return KleinSections(
        front=token_range("front"),
        mid=token_range("mid"),
        end=token_range("end"),
    )


__all__ = [
    "KLEIN_SECTIONS_KEY",
    "REFERENCE_LATENTS_KEY",
    "REFERENCE_METHOD_KEY",
    "get_klein_sections",
    "get_reference_latents",
    "latent_samples",
    "set_reference_latents",
    "split_latent_batch",
]
