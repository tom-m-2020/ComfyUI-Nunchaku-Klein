"""Direct pre-attention K/V weighting for one FLUX.2 Klein reference."""

from dataclasses import dataclass

import torch

from ...models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
)
from .common import REFERENCE_CATEGORY
from .validation import validate_finite_range, validate_int_range


@dataclass(frozen=True)
class KleinRefLatentWeightKVCallback:
    reference_index: int
    weight: float

    def __call__(self, query, key, value, metadata):
        tensors = {"query": query, "key": key, "value": value}
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor) or tensor.ndim != 4:
                shape = None if not torch.is_tensor(tensor) else list(tensor.shape)
                raise ValueError(
                    f"Direct K/V {name} must be a rank-4 tensor, got {shape}."
                )
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "Direct K/V requires identical Q/K/V shapes, got "
                f"Q={list(query.shape)}, K={list(key.shape)}, V={list(value.shape)}."
            )
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise TypeError(
                "Direct K/V requires identical Q/K/V dtypes, got "
                f"Q={query.dtype}, K={key.dtype}, V={value.dtype}."
            )
        if query.device != key.device or query.device != value.device:
            raise ValueError(
                "Direct K/V requires identical Q/K/V devices, got "
                f"Q={query.device}, K={key.device}, V={value.device}."
            )

        required_fields = (
            "block_type",
            "text_token_count",
            "generated_token_count",
            "reference_token_counts",
            "logical_image_token_count",
            "padded_text_token_count",
            "padded_image_token_count",
            "packed_sequence_length",
        )
        missing = [name for name in required_fields if not hasattr(metadata, name)]
        if missing:
            raise TypeError(
                "Direct K/V callback metadata is missing: " + ", ".join(missing)
            )

        reference_counts = metadata.reference_token_counts
        if not isinstance(reference_counts, tuple) or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in reference_counts
        ):
            raise ValueError(
                "Direct K/V reference_token_counts must be an ordered tuple "
                "of positive integers."
            )
        if not 0 <= self.reference_index < len(reference_counts):
            raise IndexError(
                "Direct K/V reference_index "
                f"{self.reference_index} is out of range for "
                f"{len(reference_counts)} runtime references."
            )

        expected_image = metadata.generated_token_count + sum(reference_counts)
        if metadata.logical_image_token_count != expected_image:
            raise ValueError(
                "Direct K/V logical image count is inconsistent: "
                f"{metadata.logical_image_token_count} != {expected_image}."
            )
        if query.shape[2] != metadata.packed_sequence_length:
            raise ValueError(
                "Direct K/V packed sequence mismatch: Q/K/V have "
                f"{query.shape[2]} tokens but metadata reports "
                f"{metadata.packed_sequence_length}."
            )
        if (
            metadata.packed_sequence_length
            != metadata.padded_text_token_count + metadata.padded_image_token_count
        ):
            raise ValueError(
                "Direct K/V padded text/image counts do not match the packed "
                "sequence length."
            )

        if metadata.block_type == "double":
            if metadata.padded_text_token_count < metadata.text_token_count:
                raise ValueError("Direct K/V double-stream text padding is invalid.")
            if metadata.padded_image_token_count < expected_image:
                raise ValueError("Direct K/V double-stream image padding is invalid.")
            image_start = metadata.padded_text_token_count
        elif metadata.block_type == "single":
            if metadata.padded_text_token_count != metadata.text_token_count:
                raise ValueError(
                    "Direct K/V single-stream text count must remain logical; "
                    "single-stream padding is trailing."
                )
            if (
                metadata.text_token_count + expected_image
                > metadata.packed_sequence_length
            ):
                raise ValueError("Direct K/V single-stream logical sequence is invalid.")
            image_start = metadata.text_token_count
        else:
            raise ValueError(
                f"Direct K/V received unsupported block_type {metadata.block_type!r}."
            )

        start = (
            image_start
            + metadata.generated_token_count
            + sum(reference_counts[: self.reference_index])
        )
        end = start + reference_counts[self.reference_index]
        logical_image_end = image_start + expected_image
        if start < image_start or end > logical_image_end or end > query.shape[2]:
            raise ValueError(
                f"Direct K/V reference range [{start}, {end}) is outside the "
                f"logical image range [{image_start}, {logical_image_end})."
            )

        if self.weight != 1.0:
            key[:, :, start:end, :].mul_(self.weight)
            value[:, :, start:end, :].mul_(self.weight)
        return None


class NunchakuKleinRefLatentWeightDirectKV:
    """Faithful pre-attention K/V multiplier for one ordered reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 7},
                ),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = (
        "Directly scales one reference's projected attention K/V tokens. "
        "Requires the callback-capable Nunchaku backend derivative."
    )

    def execute(self, model, reference_index: int, weight: float):
        reference_index = validate_int_range(
            reference_index, name="reference_index", minimum=0, maximum=7
        )
        weight = validate_finite_range(
            weight, name="weight", minimum=0.0, maximum=5.0
        )

        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Ref Latent Weight (Direct K/V) "
                "requires a MODEL from NunchakuKleinModelLoader."
            )
        transformer_options = model.model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        callbacks = transformer_options.get(ATTENTION_CALLBACKS_OPTION)
        if callbacks is None:
            callbacks = Flux2AttentionCallbacks()
        elif not isinstance(callbacks, Flux2AttentionCallbacks):
            raise TypeError(
                f"{ATTENTION_CALLBACKS_OPTION} must contain Flux2AttentionCallbacks."
            )

        callback = KleinRefLatentWeightKVCallback(
            reference_index=reference_index,
            weight=float(weight),
        )
        branch = model.clone()
        branch.model_options["transformer_options"][ATTENTION_CALLBACKS_OPTION] = (
            Flux2AttentionCallbacks(
                pre_attention_callbacks=(*callbacks.pre_attention_callbacks, callback),
                post_attention_callbacks=callbacks.post_attention_callbacks,
            )
        )
        adapter.shared_lora_state.register_patcher(branch)
        return (branch,)


__all__ = [
    "KleinRefLatentWeightKVCallback",
    "NunchakuKleinRefLatentWeightDirectKV",
]
