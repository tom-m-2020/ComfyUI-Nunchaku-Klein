"""Post-attention FLUX.2 Klein Identity Feature Transfer Final.

UI and transfer behavior follow the MIT-licensed ComfyUI-Flux2Klein-Enhancer
Final node; callback integration and strict Nunchaku layout handling are
target-owned.
"""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re
import time

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


def _evenly_spaced_indices(total, count, device):
    if count >= total:
        return torch.arange(total, device=device, dtype=torch.long)
    if count == 1:
        return torch.tensor((total // 2,), device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    return positions.mul(total - 1).floor_divide(count - 1)


def _save_debug_heatmaps(maps, grid_shape, output_directory, prefix):
    from PIL import Image

    height, width = grid_shape
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for batch_index in range(maps["confidence"].shape[0]):
        values = {
            "best_similarity": ((maps["best_similarity"][batch_index] + 1.0) * 0.5).clamp(0, 1),
            "confidence": maps["confidence"][batch_index].clamp(0, 1),
            "delta_norm": maps["delta_norm"][batch_index],
            "winning_reference_token": maps["winning_reference_token"][batch_index].float(),
        }
        delta = values["delta_norm"]
        values["delta_norm"] = (delta - delta.min()) / (delta.max() - delta.min()).clamp(min=1e-12)
        winner = values["winning_reference_token"]
        valid_winner = winner >= 0
        winner_normalized = torch.zeros_like(winner)
        if valid_winner.any():
            maximum = winner[valid_winner].max().clamp(min=1)
            winner_normalized[valid_winner] = winner[valid_winner] / maximum
        values["winning_reference_token"] = winner_normalized
        for name, value in values.items():
            pixels = (value.reshape(height, width).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            path = directory / f"{prefix}_batch{batch_index}_{name}.png"
            Image.fromarray(pixels, mode="L").save(path)
            paths.append(path)
    return tuple(paths)


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
    debug_spatial: bool = False
    debug_probe_block_type: str = "double"
    debug_probe_block_index: int = 0
    debug_output_directory: str | None = None
    debug_eligible_bank_cap: int = 0
    debug_reference_pool_shape: tuple[int, int] | None = None

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
            height, width = shapes[index]
            effective_height, effective_width = height, width
            if self.debug_reference_pool_shape is not None:
                effective_height, effective_width = self.debug_reference_pool_shape
                feature_grid = part.float().reshape(
                    part.shape[0], height, width, part.shape[2]
                ).permute(0, 3, 1, 2)
                part = F.adaptive_avg_pool2d(
                    feature_grid, (effective_height, effective_width)
                ).permute(0, 2, 3, 1).reshape(
                    part.shape[0], effective_height * effective_width, part.shape[2]
                )
                if self.debug:
                    logger.info(
                        "IFT Final diagnostic feature pool: %s %d ref=%d "
                        "source=(%d,%d) target=(%d,%d) tokens=%d->%d",
                        metadata.block_type,
                        metadata.block_index,
                        index,
                        height,
                        width,
                        effective_height,
                        effective_width,
                        height * width,
                        effective_height * effective_width,
                    )
            mask = self.masks[index] if index < len(self.masks) else None
            if mask is not None:
                pooled = F.adaptive_avg_pool2d(
                    mask[None, None], (effective_height, effective_width)
                ).flatten()
                eligible = torch.nonzero(pooled >= self.mask_threshold, as_tuple=False).flatten()
                if self.debug:
                    logger.info(
                        "IFT Final mask: %s %d ref=%d threshold=%.2f "
                        "eligible=%d/%d pooled_min=%.6f pooled_max=%.6f",
                        metadata.block_type,
                        metadata.block_index,
                        index,
                        self.mask_threshold,
                        eligible.numel(),
                        pooled.numel(),
                        float(pooled.min()),
                        float(pooled.max()),
                    )
                if eligible.numel() == 0:
                    continue
                part = part.index_select(1, eligible.to(device=attention_output.device))
            if part.shape[1]:
                bank_parts.append(part)
        if not bank_parts:
            if self.debug:
                logger.info(
                    "IFT Final: %s %d refs=%s eligible bank is empty -> no-op",
                    metadata.block_type,
                    metadata.block_index,
                    selected,
                )
            return None
        reference_bank = bank_parts[0] if len(bank_parts) == 1 else torch.cat(bank_parts, dim=1)
        eligible_tokens_before_cap = reference_bank.shape[1]
        bank_source_indices = torch.arange(
            eligible_tokens_before_cap, device=reference_bank.device, dtype=torch.long
        )
        if self.debug_eligible_bank_cap and eligible_tokens_before_cap > self.debug_eligible_bank_cap:
            selected_bank_indices = _evenly_spaced_indices(
                eligible_tokens_before_cap,
                self.debug_eligible_bank_cap,
                reference_bank.device,
            )
            reference_bank = reference_bank.index_select(1, selected_bank_indices)
            bank_source_indices = bank_source_indices.index_select(0, selected_bank_indices)
            if self.debug:
                logger.info(
                    "IFT Final diagnostic bank cap: %s %d eligible=%d capped=%d "
                    "first=%d last=%d",
                    metadata.block_type,
                    metadata.block_index,
                    eligible_tokens_before_cap,
                    reference_bank.shape[1],
                    int(selected_bank_indices[0]),
                    int(selected_bank_indices[-1]),
                )
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
        debug_count = 0
        debug_above_floor = 0
        debug_similarity_sum = 0.0
        debug_similarity_max = -math.inf
        debug_confidence_sum = 0.0
        debug_confidence_max = 0.0
        debug_delta_norm_sum = 0.0
        debug_original_norm_sum = 0.0
        collect_spatial = (
            self.debug
            and self.debug_spatial
            and metadata.block_type == self.debug_probe_block_type
            and metadata.block_index == self.debug_probe_block_index
        )
        spatial_best = []
        spatial_confidence = []
        spatial_delta_norm = []
        spatial_winner = []
        for offset in range(0, generated.shape[1], GENERATED_QUERY_CHUNK_SIZE):
            stop = min(offset + GENERATED_QUERY_CHUNK_SIZE, generated.shape[1])
            raw_similarity = torch.bmm(
                generated_normalized[:, offset:stop], reference_transposed
            )
            if self.debug:
                raw_best = raw_similarity.max(dim=-1).values
                valid_matches = raw_best >= self.similarity_floor
                debug_count += raw_best.numel()
                debug_above_floor += int(valid_matches.sum().item())
                debug_similarity_sum += float(raw_best.sum().item())
                debug_similarity_max = max(
                    debug_similarity_max, float(raw_best.max().item())
                )
                if collect_spatial:
                    winner = raw_similarity.argmax(dim=-1)
                    winner = torch.where(
                        raw_best >= self.similarity_floor,
                        bank_source_indices[winner],
                        torch.full_like(winner, -1),
                    )
                    spatial_best.append(raw_best.detach().cpu())
                    spatial_winner.append(winner.detach().cpu())
            similarity = torch.where(
                raw_similarity >= self.similarity_floor,
                raw_similarity,
                torch.full_like(raw_similarity, negative),
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
            applied_delta = (pooled.to(original.dtype) - original) * transfer_weight
            result[:, gen_start + offset : gen_start + stop] = original + applied_delta
            if self.debug:
                debug_confidence_sum += float(confidence.sum().item())
                debug_confidence_max = max(
                    debug_confidence_max, float(confidence.max().item())
                )
                debug_delta_norm_sum += float(
                    applied_delta.float().norm(dim=-1).sum().item()
                )
                debug_original_norm_sum += float(
                    original.float().norm(dim=-1).sum().item()
                )
                if collect_spatial:
                    spatial_confidence.append(confidence.detach().cpu())
                    spatial_delta_norm.append(
                        applied_delta.float().norm(dim=-1).detach().cpu()
                    )
        if self.debug:
            mean_delta_norm = debug_delta_norm_sum / max(debug_count, 1)
            relative_delta = debug_delta_norm_sum / max(
                debug_original_norm_sum, 1e-12
            )
            logger.info(
                "IFT Final: %s %d refs=%s strength=%.4f bank=%d generated=%d\n"
                "similarity: mean=%.6f max=%.6f above_floor=%d/%d (%.2f%%)\n"
                "confidence: mean=%.6f max=%.6f\n"
                "transfer: mean_delta_norm=%.6f relative_delta=%.6f",
                metadata.block_type,
                metadata.block_index,
                selected,
                strength,
                reference_bank.shape[1],
                generated.shape[1],
                debug_similarity_sum / max(debug_count, 1),
                debug_similarity_max,
                debug_above_floor,
                debug_count,
                100.0 * debug_above_floor / max(debug_count, 1),
                debug_confidence_sum / max(debug_count, 1),
                debug_confidence_max,
                mean_delta_norm,
                relative_delta,
            )
            if collect_spatial:
                generated_shape = getattr(metadata, "generated_spatial_shape", None)
                if (
                    not isinstance(generated_shape, tuple)
                    or len(generated_shape) != 2
                    or generated_shape[0] * generated_shape[1] != generated.shape[1]
                ):
                    raise ValueError(
                        "Identity Feature Transfer Final spatial diagnostics require "
                        "authoritative generated_spatial_shape metadata."
                    )
                maps = {
                    "best_similarity": torch.cat(spatial_best, dim=1),
                    "confidence": torch.cat(spatial_confidence, dim=1),
                    "delta_norm": torch.cat(spatial_delta_norm, dim=1),
                    "winning_reference_token": torch.cat(spatial_winner, dim=1),
                }
                reference_tokens_total = sum(
                    reference_ranges[index][1] - reference_ranges[index][0]
                    for index in selected
                )
                percentile_lines = []
                quantiles = torch.tensor((0.90, 0.95, 0.99), dtype=torch.float32)
                for name in ("best_similarity", "confidence", "delta_norm"):
                    values = maps[name].float().flatten()
                    p90, p95, p99 = torch.quantile(values, quantiles).tolist()
                    percentile_lines.append(
                        f"{name}: mean={float(values.mean()):.6f} "
                        f"p90={p90:.6f} p95={p95:.6f} p99={p99:.6f} "
                        f"max={float(values.max()):.6f}"
                    )
                logger.info(
                    "IFT Final probe: %s %d reference_tokens_total=%d "
                    "eligible_reference_tokens=%d matching_reference_tokens=%d\n%s",
                    metadata.block_type,
                    metadata.block_index,
                    reference_tokens_total,
                    eligible_tokens_before_cap,
                    reference_bank.shape[1],
                    "\n".join(percentile_lines),
                )
                confidence_map = maps["confidence"]
                for percentage in (1, 5, 10):
                    count = max(1, math.ceil(confidence_map.shape[1] * percentage / 100))
                    top_indices = confidence_map.topk(count, dim=1).indices
                    rows = torch.div(top_indices, generated_shape[1], rounding_mode="floor")
                    columns = top_indices.remainder(generated_shape[1])
                    box_area = (rows.max(1).values - rows.min(1).values + 1) * (
                        columns.max(1).values - columns.min(1).values + 1
                    )
                    logger.info(
                        "IFT Final spatial: top=%d%% bbox_area_fraction=%s",
                        percentage,
                        tuple(
                            round(float(value), 6)
                            for value in box_area.float().div(generated.shape[1])
                        ),
                    )
                if self.debug_output_directory is None:
                    raise RuntimeError("Identity Feature Transfer Final debug output directory is missing.")
                prefix = (
                    f"ift_final_{time.time_ns()}_{metadata.block_type}"
                    f"{metadata.block_index}"
                )
                paths = _save_debug_heatmaps(
                    maps, generated_shape, self.debug_output_directory, prefix
                )
                logger.info(
                    "IFT Final spatial diagnostics saved: %s",
                    ", ".join(str(path) for path in paths),
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
                "debug_spatial": ("BOOLEAN", {"default": False}),
                "debug_probe_block_type": (["double", "single"], {"default": "double"}),
                "debug_probe_block_index": ("INT", {"default": 0, "min": 0, "max": 23, "step": 1}),
                "debug_eligible_bank_cap": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1}),
                "debug_reference_pool_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "debug_reference_pool_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
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
        debug_spatial=False,
        debug_probe_block_type="double",
        debug_probe_block_index=0,
        debug_eligible_bank_cap=0,
        debug_reference_pool_height=0,
        debug_reference_pool_width=0,
        **mask_inputs,
    ):
        enabled = validate_bool(enabled, name="enabled")
        debug = validate_bool(debug, name="debug")
        debug_spatial = validate_bool(debug_spatial, name="debug_spatial")
        if debug_probe_block_type not in ("double", "single"):
            raise ValueError("debug_probe_block_type must be 'double' or 'single'.")
        maximum_probe_index = 7 if debug_probe_block_type == "double" else 23
        debug_probe_block_index = validate_int_range(
            debug_probe_block_index,
            name="debug_probe_block_index",
            minimum=0,
            maximum=maximum_probe_index,
        )
        debug_eligible_bank_cap = validate_int_range(
            debug_eligible_bank_cap,
            name="debug_eligible_bank_cap",
            minimum=0,
            maximum=65536,
        )
        if debug_eligible_bank_cap and not debug:
            raise ValueError("debug_eligible_bank_cap requires debug=true.")
        debug_reference_pool_height = validate_int_range(
            debug_reference_pool_height,
            name="debug_reference_pool_height",
            minimum=0,
            maximum=4096,
        )
        debug_reference_pool_width = validate_int_range(
            debug_reference_pool_width,
            name="debug_reference_pool_width",
            minimum=0,
            maximum=4096,
        )
        if bool(debug_reference_pool_height) != bool(debug_reference_pool_width):
            raise ValueError(
                "debug_reference_pool_height and debug_reference_pool_width "
                "must both be zero or both be positive."
            )
        if debug_reference_pool_height and not debug:
            raise ValueError("debug reference feature pooling requires debug=true.")
        if debug_reference_pool_height and debug_eligible_bank_cap:
            raise ValueError(
                "debug reference feature pooling cannot be combined with "
                "debug_eligible_bank_cap in one controlled experiment."
            )
        debug_reference_pool_shape = (
            (debug_reference_pool_height, debug_reference_pool_width)
            if debug_reference_pool_height
            else None
        )
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

        transformer = adapter.transformer
        double_block_count = len(getattr(transformer, "transformer_blocks", ()))
        single_block_count = len(
            getattr(transformer, "single_transformer_blocks", ())
        )
        if double_block_count <= 0 or single_block_count <= 0:
            raise RuntimeError(
                "Identity Feature Transfer Final requires positive live double "
                "and single transformer block inventories."
            )
        maximum_probe_index = (
            double_block_count - 1
            if debug_probe_block_type == "double"
            else single_block_count - 1
        )
        debug_probe_block_index = validate_int_range(
            debug_probe_block_index,
            name="debug_probe_block_index",
            minimum=0,
            maximum=maximum_probe_index,
        )
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
        debug_output_directory = None
        if debug and debug_spatial:
            import folder_paths

            debug_output_directory = str(
                Path(folder_paths.get_temp_directory())
                / "nunchaku_klein_ift_final"
            )
        callback = KleinIdentityFeatureTransferFinalCallback(
            selected,
            _parse_schedule(
                double_blocks, double_block_count, name="double_blocks"
            ),
            _parse_schedule(
                single_blocks, single_block_count, name="single_blocks"
            ),
            similarity_floor,
            softmax_temperature,
            mask_threshold,
            masks,
            debug,
            debug_spatial,
            debug_probe_block_type,
            debug_probe_block_index,
            debug_output_directory,
            debug_eligible_bank_cap,
            debug_reference_pool_shape,
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
    "_evenly_spaced_indices",
]
