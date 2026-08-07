"""Nunchaku-compatible FLUX.2 Klein Color Anchor.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

import logging
import math
from collections.abc import Mapping, Sequence

import torch

from .common import POST_CFG_CATEGORY
from .metadata import get_reference_latents
from .post_cfg import register_post_cfg_callback
from .validation import (
    validate_bchw_tensor,
    validate_bool,
    validate_conditioning,
    validate_finite_range,
    validate_int_range,
)


logger = logging.getLogger(__name__)


def _legacy_reference_latents(metadata):
    model_conditions = metadata.get("model_conds", {})
    if not isinstance(model_conditions, Mapping):
        return ()
    value = model_conditions.get("ref_latents")
    references = getattr(value, "cond", None)
    if references is None:
        return ()
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise TypeError("model_conds['ref_latents'].cond must be a sequence.")
    return tuple(
        validate_bchw_tensor(reference, name=f"model_conds ref_latents[{index}]")
        for index, reference in enumerate(references)
    )


def _selected_reference(conditioning, ref_index):
    validate_conditioning(conditioning)
    for item in conditioning:
        references = get_reference_latents(item[1])
        if not references:
            references = _legacy_reference_latents(item[1])
        if ref_index < len(references):
            return references[ref_index]
    return None


def _channel_statistics(reference, channel_weights):
    working = reference.detach().to(device="cpu", dtype=torch.float32)
    means = working.mean(dim=(-2, -1), keepdim=True)
    trust = None
    if channel_weights == "by_variance":
        spatial_variance = working.var(dim=(-2, -1), keepdim=True)
        trust = 1.0 / (1.0 + spatial_variance)
        trust = trust / trust.max().clamp(min=1e-8)

    mean_values = tuple(tuple(float(value) for value in row) for row in means[:, :, 0, 0])
    trust_values = None
    if trust is not None:
        trust_values = tuple(
            tuple(float(value) for value in row) for row in trust[:, :, 0, 0]
        )
    return mean_values, trust_values


class NunchakuKleinColorAnchor:
    """Correct denoised channel means through ComfyUI's post-CFG boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Maximum correction strength. "
                            "0.3-0.6 is a good starting range. "
                            "Too high and you override the model's color decisions entirely."
                        ),
                    },
                ),
            },
            "optional": {
                "ramp_curve": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": (
                            "Controls the shape of the correction ramp. "
                            "Formula: progress^(1/curve). "
                            "1.0 = linear. "
                            ">1 = fast start, tapers off  (e.g. sqrt for curve=2). "
                            "<1 = slow start, aggressive late  (e.g. squared for curve=0.5). "
                            "For few-step schedules (4-8 steps) values of 2-4 work well "
                            "because they reach useful strength quickly."
                        ),
                    },
                ),
                "ref_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 63,
                        "tooltip": "Which reference latent to anchor colors from.",
                    },
                ),
                "channel_weights": (
                    ["uniform", "by_variance"],
                    {
                        "default": "uniform",
                        "tooltip": (
                            "uniform: correct all channels equally. "
                            "by_variance: weight correction by how stable each channel's "
                            "mean is in the reference (low-variance channels trusted more)."
                        ),
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = POST_CFG_CATEGORY

    def apply(
        self,
        model,
        conditioning,
        strength=0.5,
        ramp_curve=1.5,
        ref_index=0,
        channel_weights="uniform",
        debug=False,
    ):
        strength = validate_finite_range(
            strength, name="strength", minimum=0.0, maximum=1.0
        )
        if strength == 0.0:
            return (model,)

        ramp_curve = validate_finite_range(
            ramp_curve, name="ramp_curve", minimum=0.5, maximum=8.0
        )
        ref_index = validate_int_range(
            ref_index, name="ref_index", minimum=0, maximum=63
        )
        if channel_weights not in ("uniform", "by_variance"):
            raise ValueError(
                "channel_weights must be 'uniform' or 'by_variance', "
                f"got {channel_weights!r}."
            )
        debug = validate_bool(debug, name="debug")

        reference = _selected_reference(conditioning, ref_index)
        if reference is None:
            logger.warning(
                "Color Anchor found no reference latent at ref_index=%d; node inactive.",
                ref_index,
            )
            return (model,)

        reference_means, channel_trust = _channel_statistics(
            reference, channel_weights
        )
        reference_batch = len(reference_means)
        reference_channels = len(reference_means[0])
        state = {
            "sample_options_id": None,
            "sigma_max": None,
            "last_sigma": None,
            "last_sigma_logged": None,
            "step": 0,
        }

        def color_anchor(args):
            if not isinstance(args, Mapping):
                raise TypeError("Color Anchor post-CFG arguments must be a mapping.")
            denoised = args.get("denoised")
            sigma = args.get("sigma")
            if not torch.is_tensor(denoised) or denoised.ndim != 4:
                shape = getattr(denoised, "shape", None)
                raise ValueError(
                    "Color Anchor requires BCHW args['denoised']; "
                    f"got {shape}."
                )
            if not torch.is_tensor(sigma) or sigma.numel() == 0:
                raise TypeError("Color Anchor requires a non-empty tensor args['sigma'].")
            if denoised.shape[1] != reference_channels:
                raise ValueError(
                    "Color Anchor channel mismatch: denoised has "
                    f"{denoised.shape[1]} channels, reference has {reference_channels}."
                )
            if reference_batch not in (1, denoised.shape[0]):
                raise ValueError(
                    "Color Anchor reference batch must be 1 or match denoised batch; "
                    f"got reference={reference_batch}, denoised={denoised.shape[0]}."
                )

            sigma_value = float(sigma.max().item())
            model_options = args.get("model_options")
            if not isinstance(model_options, Mapping):
                raise TypeError(
                    "Color Anchor requires mapping args['model_options']."
                )
            # ComfyUI creates one model-options clone per sampler execution.
            # Its identity lets a cached branch reset scalar progress even when
            # a new one-step run begins at the same sigma as the previous run.
            sample_options_id = id(model_options)
            if sample_options_id != state["sample_options_id"]:
                state["sample_options_id"] = sample_options_id
                state["sigma_max"] = None
                state["last_sigma"] = None
                state["step"] = 0

            sigma_max = state["sigma_max"]
            last_sigma = state["last_sigma"]
            if (
                sigma_max is None
                or sigma_value > sigma_max
                or (last_sigma is not None and sigma_value > last_sigma and sigma_value >= sigma_max)
            ):
                state["sigma_max"] = sigma_value
                state["step"] = 0
            state["last_sigma"] = sigma_value

            sigma_max = state["sigma_max"]
            sigma_progress = max(
                0.0,
                min(
                    1.0,
                    (sigma_max - sigma_value) / sigma_max
                    if sigma_max > 1e-6
                    else 0.0,
                ),
            )
            state["step"] += 1
            step_progress = 1.0 - 0.5 ** state["step"]
            progress = max(sigma_progress, step_progress)
            effective_strength = strength * progress ** (1.0 / ramp_curve)
            if effective_strength < 1e-5:
                return denoised

            reference_mean = torch.tensor(
                reference_means, device=denoised.device, dtype=denoised.dtype
            )[:, :, None, None]
            current_mean = denoised.mean(dim=(-2, -1), keepdim=True)
            correction = reference_mean - current_mean
            if channel_trust is not None:
                trust = torch.tensor(
                    channel_trust, device=denoised.device, dtype=denoised.dtype
                )[:, :, None, None]
                correction = correction * trust
            corrected = denoised + correction * effective_strength

            if debug and sigma_value != state["last_sigma_logged"]:
                state["last_sigma_logged"] = sigma_value
                logger.info(
                    "Color Anchor step=%d sigma=%.4f sigma_progress=%.3f "
                    "step_progress=%.3f progress=%.3f effective=%.3f "
                    "mean_drift=%.5f applied=%.5f",
                    state["step"],
                    sigma_value,
                    sigma_progress,
                    step_progress,
                    progress,
                    effective_strength,
                    float((reference_mean - current_mean).abs().mean()),
                    float((correction * effective_strength).abs().mean()),
                )
            return corrected

        return (register_post_cfg_callback(model, color_anchor),)


__all__ = ["NunchakuKleinColorAnchor"]
