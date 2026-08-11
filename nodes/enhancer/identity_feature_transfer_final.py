"""Post-attention FLUX.2 Klein Identity Feature Transfer Final.

UI and transfer behavior follow the MIT-licensed ComfyUI-Flux2Klein-Enhancer
Final node; callback integration and strict Nunchaku layout handling are
target-owned.
"""

from dataclasses import dataclass
import logging
import math
import re

import torch
from torch.nn import functional as F

from ...models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    IDENTITY_FEATURE_TRANSFER_FINAL_OPTION,
    NunchakuFlux2KleinAdapter,
)
from .attention_kv import resolve_attention_token_ranges
from .common import REFERENCE_CATEGORY
from .validation import validate_bool, validate_finite_range, validate_int_range, validate_mask


logger = logging.getLogger(__name__)
HARD_DOUBLE = "0-7:mid_img=0.55"
HARD_SINGLE = (
    "0:mid_img=0.22; 1:mid_img=0.24; 3:mid_img=0.28; "
    "4:mid_img=0.22; 6:mid_img=0.26; 7:mid_img=0.27; "
    "8:mid_img=0.25; 10:mid_img=0.27; 13:mid_img=0.27"
)
PRESETS = {
    "HARD_LOCK": (HARD_DOUBLE, HARD_SINGLE, 0.040, 0.0250, 1.0),
    "MID_LOCK": (HARD_DOUBLE, HARD_SINGLE, 0.200, 0.0700, 1.0),
    "SOFT_LOCK": (HARD_DOUBLE, HARD_SINGLE, 0.500, 0.0700, 1.0),
}
GENERATED_QUERY_CHUNK_SIZE = 256


def _parse_schedule(text, block_count, *, name):
    resolved = [0.0] * block_count
    for row in str(text or "").split(";"):
        row = row.strip()
        if not row or ":" not in row:
            continue
        block_part, value_part = row.split(":", 1)
        if "=" in value_part:
            key, value_part = value_part.split("=", 1)
            if key.strip().lower() not in ("mid", "mid_img"):
                continue
        try:
            strength = float(value_part.strip())
            if not math.isfinite(strength):
                raise ValueError
            if "-" in block_part:
                first, last = (int(value.strip()) for value in block_part.split("-", 1))
            else:
                first = last = int(block_part.strip())
        except ValueError as error:
            raise ValueError(f"{name} contains an invalid schedule row: {row!r}.") from error
        first, last = sorted((first, last))
        for index in range(max(0, first), min(block_count - 1, last) + 1):
            resolved[index] = strength
    return tuple(resolved)


def _parse_reference_selection(text, fallback):
    value = str(text or "all").strip().lower()
    if value in ("", "all", "*"):
        return None
    selected = []
    for part in re.split(r"[;, ]+", value):
        if not part:
            continue
        try:
            if "-" in part:
                first, last = (int(item) for item in part.split("-", 1))
                step = 1 if last >= first else -1
                selected.extend(range(first, last + step, step))
            else:
                selected.append(int(part))
        except ValueError as error:
            raise ValueError(f"reference_indices contains an invalid entry: {part!r}.") from error
    if not selected:
        selected = [fallback]
    if any(index < 0 or index > 15 for index in selected):
        raise ValueError("reference_indices entries must be within [0, 15].")
    return tuple(selected)


def _prepare_mask(mask, *, name):
    if mask is None:
        return None
    validate_mask(mask)
    value = mask.detach().float().cpu()
    if value.ndim == 4:
        value = value[0].mean(dim=-1) if value.shape[-1] in (1, 3, 4) else value[0, 0]
    elif value.ndim == 3:
        value = value.mean(dim=-1) if value.shape[-1] in (1, 3, 4) and value.shape[0] != 1 else value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} cannot be reduced to a two-dimensional mask.")
    return value.clone().contiguous()


@dataclass(frozen=True)
class KleinIdentityFeatureTransferFinalCallback:
    selected_reference_indices: tuple[int, ...] | None
    double_strengths: tuple[float, ...]
    single_strengths: tuple[float, ...]
    similarity_floor: float
    softmax_temperature: float
    mask_threshold: float
    masks: tuple[torch.Tensor | None, ...]
    debug: bool = False

    def __call__(self, attention_output, metadata):
        if not torch.is_tensor(attention_output) or attention_output.ndim != 3:
            shape = None if not torch.is_tensor(attention_output) else list(attention_output.shape)
            raise ValueError(f"Identity Feature Transfer Final output must be rank 3, got {shape}.")
        if not attention_output.is_floating_point():
            raise TypeError("Identity Feature Transfer Final output must be floating point.")
        if not attention_output.is_contiguous():
            raise ValueError("Identity Feature Transfer Final output must be contiguous.")
        if attention_output.shape[0] != getattr(metadata, "batch_size", None):
            raise ValueError(
                "Identity Feature Transfer Final output batch does not match callback metadata."
            )
        expected_features = getattr(metadata, "head_count", 0) * getattr(
            metadata, "head_dimension", 0
        )
        if attention_output.shape[2] != expected_features:
            raise ValueError(
                "Identity Feature Transfer Final output width does not match "
                "head_count * head_dimension."
            )
        _, generated_range, reference_ranges = resolve_attention_token_ranges(
            metadata, attention_output.shape[1], name="Identity Feature Transfer Final"
        )
        shapes = getattr(metadata, "reference_spatial_shapes", None)
        if not isinstance(shapes, tuple) or len(shapes) != len(reference_ranges):
            raise ValueError(
                "Identity Feature Transfer Final requires one immutable "
                "reference_spatial_shapes entry per runtime reference."
            )
        for index, (shape, token_range) in enumerate(zip(shapes, reference_ranges, strict=True)):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
                or shape[0] * shape[1] != token_range[1] - token_range[0]
            ):
                raise ValueError(
                    f"reference_spatial_shapes[{index}] must match its positive token grid."
                )

        if metadata.block_type == "double":
            strengths = self.double_strengths
        elif metadata.block_type == "single":
            strengths = self.single_strengths
        else:
            raise ValueError(f"Unsupported block_type {metadata.block_type!r}.")
        if not 0 <= metadata.block_index < len(strengths):
            raise IndexError(
                f"Identity Feature Transfer Final block index {metadata.block_index} "
                f"is invalid for {metadata.block_type} blocks."
            )
        strength = strengths[metadata.block_index]
        if strength <= 0.0:
            return None

        if self.selected_reference_indices is None:
            selected = tuple(range(len(reference_ranges)))
        else:
            selected = self.selected_reference_indices
            for index in selected:
                if index >= len(reference_ranges):
                    raise IndexError(
                        "Identity Feature Transfer Final reference index "
                        f"{index} is out of range for {len(reference_ranges)} runtime references."
                    )
        if not selected:
            return None

        bank_parts = []
        for index in selected:
            start, end = reference_ranges[index]
            part = attention_output[:, start:end]
            mask = self.masks[index] if index < len(self.masks) else None
            if mask is not None:
                height, width = shapes[index]
                pooled = F.adaptive_avg_pool2d(mask[None, None], (height, width)).flatten()
                eligible = torch.nonzero(pooled >= self.mask_threshold, as_tuple=False).flatten()
                if eligible.numel() == 0:
                    continue
                part = part.index_select(1, eligible.to(device=attention_output.device))
            if part.shape[1]:
                bank_parts.append(part)
        if not bank_parts:
            return None
        reference_bank = bank_parts[0] if len(bank_parts) == 1 else torch.cat(bank_parts, dim=1)
        gen_start, gen_end = generated_range
        generated = attention_output[:, gen_start:gen_end]
        if generated.shape[1] == 0:
            return None

        generated_float = generated.float()
        reference_float = reference_bank.float()
        generated_normalized = F.normalize(
            generated_float - generated_float.mean(dim=1, keepdim=True), dim=-1
        )
        reference_normalized = F.normalize(
            reference_float - reference_float.mean(dim=1, keepdim=True), dim=-1
        )
        reference_transposed = reference_normalized.transpose(1, 2)
        result = attention_output.clone()
        negative = torch.finfo(torch.float32).min
        confidence_denominator = max(1.0 - self.similarity_floor, 1e-6)
        for offset in range(0, generated.shape[1], GENERATED_QUERY_CHUNK_SIZE):
            stop = min(offset + GENERATED_QUERY_CHUNK_SIZE, generated.shape[1])
            similarity = torch.bmm(
                generated_normalized[:, offset:stop], reference_transposed
            )
            similarity = torch.where(
                similarity >= self.similarity_floor,
                similarity,
                torch.full_like(similarity, negative),
            )
            weights = torch.softmax(similarity / self.softmax_temperature, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            pooled = torch.bmm(weights, reference_float)
            best = similarity.max(dim=-1).values
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
            confidence = (
                (best - self.similarity_floor) / confidence_denominator
            ).clamp(0.0, 1.0)
            original = generated[:, offset:stop]
            transfer_weight = (confidence * strength).unsqueeze(-1).to(original.dtype)
            result[:, gen_start + offset : gen_start + stop] = original + (
                pooled.to(original.dtype) - original
            ) * transfer_weight
        if self.debug:
            logger.info(
                "Identity Feature Transfer Final: %s %d refs=%s strength=%.4f bank=%d",
                metadata.block_type,
                metadata.block_index,
                selected,
                strength,
                reference_bank.shape[1],
            )
        return result


class NunchakuKleinIdentityFeatureTransferFinal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (["HARD_LOCK", "MID_LOCK", "SOFT_LOCK", "custom"], {"default": "HARD_LOCK"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False}),
                "similarity_floor": ("FLOAT", {"default": 0.040, "min": 0.0, "max": 0.95, "step": 0.001}),
                "softmax_temperature": ("FLOAT", {"default": 0.0250, "min": 0.0001, "max": 0.25, "step": 0.0001}),
                "mask_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks": ("STRING", {"default": HARD_DOUBLE, "multiline": False}),
                "single_blocks": ("STRING", {"default": HARD_SINGLE, "multiline": False}),
                "debug": ("BOOLEAN", {"default": False}),
                "mask_behavior": (["focus_only", "zero_unmasked_tokens"], {
                    "default": "focus_only",
                    "tooltip": "focus_only preserves the original masking behavior: the mask limits this node's reference bank while Klein still sees the complete reference. zero_unmasked_tokens blocks each wired reference's unmasked tokens as attention sources in every block. References without a wired mask remain complete and unchanged.",
                }),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"forceInput": True, "tooltip": "Sigma-aware strength scheduling is not supported by this first Nunchaku slice."}),
                **{f"subject_mask_{index}": ("MASK",) for index in range(1, 9)},
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = "Transfers identity features from a global reference bank after attention."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    def apply(
        self,
        model,
        preset="HARD_LOCK",
        enabled=True,
        reference_index=0,
        reference_indices="all",
        similarity_floor=0.040,
        softmax_temperature=0.0250,
        mask_threshold=1.0,
        double_blocks=HARD_DOUBLE,
        single_blocks=HARD_SINGLE,
        debug=False,
        mask_behavior="focus_only",
        sigmas=None,
        **mask_inputs,
    ):
        enabled = validate_bool(enabled, name="enabled")
        debug = validate_bool(debug, name="debug")
        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Identity Feature Transfer (Final) "
                "requires a MODEL from NunchakuKleinModelLoader."
            )
        branch = model.clone()
        adapter.shared_lora_state.register_patcher(branch)
        if not enabled:
            return (branch,)

        reference_index = validate_int_range(reference_index, name="reference_index", minimum=0, maximum=15)
        if preset not in (*PRESETS, "custom"):
            raise ValueError(f"Unsupported preset {preset!r}.")
        if mask_behavior not in ("focus_only", "zero_unmasked_tokens"):
            raise ValueError(f"Unsupported mask_behavior {mask_behavior!r}.")
        if mask_behavior == "zero_unmasked_tokens":
            raise NotImplementedError(
                "Identity Feature Transfer Final zero_unmasked_tokens requires "
                "attention-source exclusion before attention_fp16 softmax; the "
                "current Nunchaku backend does not expose a mask-capable path."
            )
        if sigmas is not None:
            raise NotImplementedError(
                "Identity Feature Transfer Final sigma scheduling requires current "
                "sampling sigma in generic attention callback metadata."
            )

        if preset in PRESETS:
            double_blocks, single_blocks, similarity_floor, softmax_temperature, mask_threshold = PRESETS[preset]
        similarity_floor = validate_finite_range(similarity_floor, name="similarity_floor", minimum=0.0, maximum=0.95)
        softmax_temperature = validate_finite_range(softmax_temperature, name="softmax_temperature", minimum=0.0001, maximum=0.25)
        mask_threshold = validate_finite_range(mask_threshold, name="mask_threshold", minimum=0.0, maximum=1.0)
        selected = _parse_reference_selection(reference_indices, reference_index)
        masks = tuple(
            _prepare_mask(mask_inputs.get(f"subject_mask_{index}"), name=f"subject_mask_{index}")
            for index in range(1, 9)
        )
        callback = KleinIdentityFeatureTransferFinalCallback(
            selected,
            _parse_schedule(double_blocks, 8, name="double_blocks"),
            _parse_schedule(single_blocks, 24, name="single_blocks"),
            similarity_floor,
            softmax_temperature,
            mask_threshold,
            masks,
            debug,
        )
        if not any(strength > 0.0 for strength in (*callback.double_strengths, *callback.single_strengths)):
            return (branch,)
        options = branch.model_options.get("transformer_options")
        if not isinstance(options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        callbacks = options.get(ATTENTION_CALLBACKS_OPTION, Flux2AttentionCallbacks())
        if not isinstance(callbacks, Flux2AttentionCallbacks):
            raise TypeError(f"{ATTENTION_CALLBACKS_OPTION} must contain Flux2AttentionCallbacks.")
        inherited = options.get(IDENTITY_FEATURE_TRANSFER_FINAL_OPTION, ())
        if not isinstance(inherited, tuple) or not all(callable(item) for item in inherited):
            raise TypeError(f"{IDENTITY_FEATURE_TRANSFER_FINAL_OPTION} must contain an immutable tuple of callables.")
        options[ATTENTION_CALLBACKS_OPTION] = Flux2AttentionCallbacks(
            callbacks.pre_attention_callbacks,
            (*callbacks.post_attention_callbacks, callback),
        )
        options[IDENTITY_FEATURE_TRANSFER_FINAL_OPTION] = (*inherited, callback)
        return (branch,)


__all__ = [
    "KleinIdentityFeatureTransferFinalCallback",
    "NunchakuKleinIdentityFeatureTransferFinal",
]
