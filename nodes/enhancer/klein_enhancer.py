"""Nunchaku-compatible FLUX.2 Klein conditioning enhancer.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

import logging

import torch

import comfy.model_management as model_management

from .common import TEXT_CATEGORY
from .conditioning import clone_conditioning, restore_dtype, temporary_float32
from .validation import validate_bool, validate_finite_range, validate_int_range


logger = logging.getLogger(__name__)

_LAYER_WIDTHS = {12288: 4096, 7680: 2560}


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


class NunchakuKleinEnhancer:
    """Apply explicit transforms to active Klein conditioning tokens."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "active_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Multiplier on every active-token embedding. 1.0 = unchanged. The model was trained on Qwen3's natural distribution; values far from 1.0 push it off-distribution.",
                    },
                ),
                "per_token_whiten": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Amplifies per-token deviation from the sequence mean: (x - mean)*(1+w) + mean. >0 widens spread, <0 compresses. Was called 'contrast' in v1.",
                    },
                ),
                "norm_equalize": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend each token toward the per-sequence mean L2 norm. Flattens magnitude variance — fights Qwen3's natural emphasis. 0 = no effect.",
                    },
                ),
            },
            "optional": {
                "early_layer_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Klein-specific. Scale the first Qwen3 layer slice (low-level / structural features). Klein conditioning stacks 3 layers along the embed dim; this targets the first.",
                    },
                ),
                "mid_layer_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Klein-specific. Scale the middle Qwen3 layer slice (intermediate semantic features).",
                    },
                ),
                "late_layer_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Klein-specific. Scale the last Qwen3 layer slice (high-level / abstract semantic features).",
                    },
                ),
                "preserve_original": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Linear blend back the unmodified active region. 0.0 = full enhancement, 1.0 = no change.",
                    },
                ),
                "active_end_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Override the active-region end. 0 = auto-detect from attention_mask, falls back to full sequence length if mask missing.",
                    },
                ),
                "device": (_available_devices(), {"default": "auto"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = TEXT_CATEGORY
    DESCRIPTION = "Adjusts active FLUX.2 Klein conditioning embeddings."

    def enhance(
        self,
        conditioning,
        active_scale=1.0,
        per_token_whiten=0.0,
        norm_equalize=0.0,
        early_layer_scale=1.0,
        mid_layer_scale=1.0,
        late_layer_scale=1.0,
        preserve_original=0.0,
        active_end_override=0,
        device="auto",
        debug=False,
    ):
        active_scale = validate_finite_range(
            active_scale, name="active_scale", minimum=0.0, maximum=10.0
        )
        per_token_whiten = validate_finite_range(
            per_token_whiten,
            name="per_token_whiten",
            minimum=-1.0,
            maximum=5.0,
        )
        norm_equalize = validate_finite_range(
            norm_equalize, name="norm_equalize", minimum=0.0, maximum=1.0
        )
        early_layer_scale = validate_finite_range(
            early_layer_scale,
            name="early_layer_scale",
            minimum=0.0,
            maximum=5.0,
        )
        mid_layer_scale = validate_finite_range(
            mid_layer_scale,
            name="mid_layer_scale",
            minimum=0.0,
            maximum=5.0,
        )
        late_layer_scale = validate_finite_range(
            late_layer_scale,
            name="late_layer_scale",
            minimum=0.0,
            maximum=5.0,
        )
        preserve_original = validate_finite_range(
            preserve_original,
            name="preserve_original",
            minimum=0.0,
            maximum=1.0,
        )
        active_end_override = validate_int_range(
            active_end_override,
            name="active_end_override",
            minimum=0,
            maximum=512,
        )
        debug = validate_bool(debug, name="debug")

        neutral = (
            active_scale == 1.0
            and per_token_whiten == 0.0
            and norm_equalize == 0.0
            and early_layer_scale == 1.0
            and mid_layer_scale == 1.0
            and late_layer_scale == 1.0
            and preserve_original == 0.0
        )
        if not conditioning or neutral:
            return (conditioning,)

        compute_device = _resolve_device(device)
        output = clone_conditioning(conditioning)
        for index, item in enumerate(output):
            source = item[0]
            if source.ndim != 3:
                raise ValueError(
                    "FLUX.2 Klein conditioning must have shape "
                    f"[batch, tokens, embedding], got {list(source.shape)}."
                )

            embedding_width = source.shape[2]
            layer_width = _LAYER_WIDTHS.get(embedding_width)
            if layer_width is None:
                raise ValueError(
                    "Unsupported FLUX.2 Klein conditioning width "
                    f"{embedding_width}; expected 12288 (9B) or 7680 (smaller Klein)."
                )

            working, original_dtype = temporary_float32(
                source.to(compute_device)
            )
            sequence_length = source.shape[1]
            if active_end_override > 0:
                active_end = min(active_end_override, sequence_length)
            else:
                attention_mask = item[1].get("attention_mask")
                if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
                    nonzero = attention_mask[0].nonzero()
                    active_end = (
                        int(nonzero[-1].item()) + 1
                        if len(nonzero)
                        else sequence_length
                    )
                else:
                    active_end = sequence_length

            active = working[:, :active_end, :].clone()
            original_active = active.clone()

            if per_token_whiten != 0.0 and active.numel() > 0:
                sequence_mean = active.mean(dim=1, keepdim=True)
                active = sequence_mean + (active - sequence_mean) * (
                    1.0 + per_token_whiten
                )

            if norm_equalize > 0.0 and active.numel() > 0:
                token_norms = active.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                target_norm = token_norms.mean()
                normalized = active / token_norms * target_norm
                active = (
                    active * (1.0 - norm_equalize)
                    + normalized * norm_equalize
                )

            if active_scale != 1.0:
                active = active * active_scale

            if early_layer_scale != 1.0:
                active[:, :, :layer_width] *= early_layer_scale
            if mid_layer_scale != 1.0:
                active[:, :, layer_width : 2 * layer_width] *= mid_layer_scale
            if late_layer_scale != 1.0:
                active[:, :, 2 * layer_width :] *= late_layer_scale

            if preserve_original > 0.0:
                active = (
                    active * (1.0 - preserve_original)
                    + original_active * preserve_original
                )

            result = source.detach().to("cpu").clone()
            result[:, :active_end, :] = restore_dtype(
                active, original_dtype
            ).to("cpu")
            item[0] = result

            if debug:
                difference = (
                    result[:, :active_end].float()
                    - source[:, :active_end].detach().to("cpu").float()
                ).abs()
                mean_difference = float(difference.mean()) if difference.numel() else 0.0
                max_difference = float(difference.max()) if difference.numel() else 0.0
                logger.info(
                    "Klein Enhancer item %d: shape=%s active=[0:%d] "
                    "layer_width=%d mean_diff=%.6f max_diff=%.6f",
                    index,
                    list(source.shape),
                    active_end,
                    layer_width,
                    mean_difference,
                    max_difference,
                )

        return (output,)


__all__ = ["NunchakuKleinEnhancer"]
