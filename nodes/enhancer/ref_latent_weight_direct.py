"""Direct pre-attention K/V weighting for one FLUX.2 Klein reference."""

from dataclasses import dataclass

from ...models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
)
from .attention_kv import validate_attention_kv_layout
from .common import REFERENCE_CATEGORY
from .validation import validate_finite_range, validate_int_range


@dataclass(frozen=True)
class KleinRefLatentWeightKVCallback:
    reference_index: int
    weight: float

    def __call__(self, query, key, value, metadata):
        _, _, reference_ranges = validate_attention_kv_layout(
            query, key, value, metadata, name="Direct K/V Ref Latent Weight"
        )
        if not 0 <= self.reference_index < len(reference_ranges):
            raise IndexError(
                "Direct K/V reference_index "
                f"{self.reference_index} is out of range for "
                f"{len(reference_ranges)} runtime references."
            )
        start, end = reference_ranges[self.reference_index]

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
