"""Direct pre-attention text/reference K/V balance for FLUX.2 Klein."""

from dataclasses import dataclass
import logging

from ...models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_WEIGHT_OPTION,
    TEXT_REF_BALANCE_DIRECT_OPTION,
    TEXT_REF_BALANCE_OPTION,
)
from .attention_kv import validate_attention_kv_layout
from .common import REFERENCE_CATEGORY
from .validation import (
    validate_bool,
    validate_conditioning,
    validate_finite_range,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KleinTextRefBalanceKVCallback:
    text_scale: float
    reference_scale: float
    debug: bool = False

    def __call__(self, query, key, value, metadata):
        text_range, _, reference_ranges = validate_attention_kv_layout(
            query, key, value, metadata, name="Direct K/V Text/Ref Balance"
        )
        if self.text_scale != 1.0 and text_range[0] != text_range[1]:
            start, end = text_range
            key[:, :, start:end, :].mul_(self.text_scale)
            value[:, :, start:end, :].mul_(self.text_scale)
        if self.reference_scale != 1.0:
            for start, end in reference_ranges:
                key[:, :, start:end, :].mul_(self.reference_scale)
                value[:, :, start:end, :].mul_(self.reference_scale)
        if self.debug:
            logger.info(
                "Text/Ref Balance Direct K/V: %s %d text_scale=%.3f "
                "reference_scale=%.3f references=%d",
                metadata.block_type,
                metadata.block_index,
                self.text_scale,
                self.reference_scale,
                len(reference_ranges),
            )
        return None


class NunchakuKleinTextRefBalanceDirectKV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": (
                    "FLOAT",
                    {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.001},
                ),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance_streams"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = (
        "Directly attenuates projected text or reference K/V around a neutral "
        "0.5 midpoint. Requires the callback-capable Nunchaku backend derivative."
    )

    def balance_streams(self, model, conditioning, balance=0.5, debug=False):
        validate_conditioning(conditioning)
        balance = validate_finite_range(
            balance, name="balance", minimum=0.0, maximum=1.0
        )
        debug = validate_bool(debug, name="debug")
        if balance <= 0.5:
            text_scale = 2.0 * balance
            reference_scale = 1.0
        else:
            text_scale = 1.0
            reference_scale = 2.0 * (1.0 - balance)

        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Text/Ref Balance (Direct K/V) "
                "requires a MODEL from NunchakuKleinModelLoader."
            )
        transformer_options = model.model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        if (
            transformer_options.get(REF_LATENT_WEIGHT_OPTION) is not None
            or transformer_options.get(TEXT_REF_BALANCE_OPTION) is not None
        ):
            raise ValueError(
                "Nunchaku FLUX.2 Klein Text/Ref Balance (Direct K/V) cannot "
                "be combined with prediction-space Ref Latent Weight or "
                "Text/Ref Balance. Remove one algorithm family."
            )
        callbacks = transformer_options.get(ATTENTION_CALLBACKS_OPTION)
        if callbacks is None:
            callbacks = Flux2AttentionCallbacks()
        elif not isinstance(callbacks, Flux2AttentionCallbacks):
            raise TypeError(
                f"{ATTENTION_CALLBACKS_OPTION} must contain Flux2AttentionCallbacks."
            )
        inherited_direct = transformer_options.get(
            TEXT_REF_BALANCE_DIRECT_OPTION, ()
        )
        if not isinstance(inherited_direct, tuple) or not all(
            callable(callback) for callback in inherited_direct
        ):
            raise TypeError(
                f"{TEXT_REF_BALANCE_DIRECT_OPTION} must contain an immutable "
                "tuple of callables."
            )

        callback = KleinTextRefBalanceKVCallback(
            text_scale=float(text_scale),
            reference_scale=float(reference_scale),
            debug=debug,
        )
        if debug:
            logger.info(
                "Text/Ref Balance Direct K/V: balance=%.3f text_scale=%.3f "
                "reference_scale=%.3f",
                balance,
                text_scale,
                reference_scale,
            )
        branch = model.clone()
        branch_options = branch.model_options["transformer_options"]
        branch_options[ATTENTION_CALLBACKS_OPTION] = Flux2AttentionCallbacks(
            pre_attention_callbacks=(*callbacks.pre_attention_callbacks, callback),
            post_attention_callbacks=callbacks.post_attention_callbacks,
        )
        branch_options[TEXT_REF_BALANCE_DIRECT_OPTION] = (
            *inherited_direct,
            callback,
        )
        adapter.shared_lora_state.register_patcher(branch)
        return branch, conditioning


__all__ = [
    "KleinTextRefBalanceKVCallback",
    "NunchakuKleinTextRefBalanceDirectKV",
]
