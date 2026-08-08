from ...models.klein_wrapper import (
    KleinTextRefBalanceSpec,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_WEIGHT_OPTION,
    TEXT_REF_BALANCE_OPTION,
)
from .common import REFERENCE_CATEGORY
from .validation import (
    validate_bool,
    validate_conditioning,
    validate_finite_range,
)


class NunchakuKleinTextRefBalance:
    """Prediction-space substitute for the original text/reference K/V node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": (
                    "FLOAT",
                    {"default": 0.500, "min": 0.000, "max": 1.000, "step": 0.005},
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
        "Balances text against all references using prediction proxies. This "
        "is Nunchaku compatibility behavior, not attention K/V equivalence."
    )

    def balance_streams(self, model, conditioning, balance=0.5, debug=False):
        validate_conditioning(conditioning)
        balance = validate_finite_range(
            balance,
            name="balance",
            minimum=0.0,
            maximum=1.0,
        )
        debug = validate_bool(debug, name="debug")

        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Text/Ref Balance requires a MODEL "
                "from NunchakuKleinModelLoader."
            )
        transformer_options = model.model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        if transformer_options.get(TEXT_REF_BALANCE_OPTION) is not None:
            raise ValueError(
                "Only one active Nunchaku FLUX.2 Klein Text/Ref Balance "
                "specification is supported per MODEL branch."
            )
        if transformer_options.get(REF_LATENT_WEIGHT_OPTION) is not None:
            raise ValueError(
                "Nunchaku FLUX.2 Klein Text/Ref Balance cannot be stacked "
                "with Ref Latent Weight in this release. Remove one node; "
                "prediction-residual composition is not defined."
            )

        branch = model.clone()
        branch.model_options["transformer_options"][TEXT_REF_BALANCE_OPTION] = (
            KleinTextRefBalanceSpec(balance=balance, debug=debug)
        )
        adapter.shared_lora_state.register_patcher(branch)
        return (branch, conditioning)
