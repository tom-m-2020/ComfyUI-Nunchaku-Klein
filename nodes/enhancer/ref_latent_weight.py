from ...models.klein_wrapper import (
    KleinRefLatentWeightSpec,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_WEIGHT_OPTION,
    REF_LATENT_CONTROLLER_DIRECT_OPTION,
    TEXT_REF_BALANCE_OPTION,
)
from .common import REFERENCE_CATEGORY
from .validation import validate_finite_range, validate_int_range


class NunchakuKleinRefLatentWeight:
    """Prediction-residual substitute for the original attention K/V node."""

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
        "Controls one reference through prediction residuals. This is a "
        "Nunchaku compatibility substitute, not attention K/V scaling."
    )

    def execute(self, model, reference_index: int, weight: float):
        reference_index = validate_int_range(
            reference_index,
            name="reference_index",
            minimum=0,
            maximum=7,
        )
        weight = validate_finite_range(
            weight,
            name="weight",
            minimum=0.0,
            maximum=5.0,
        )

        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Ref Latent Weight requires a MODEL "
                "from NunchakuKleinModelLoader."
            )
        transformer_options = model.model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        if transformer_options.get(REF_LATENT_WEIGHT_OPTION) is not None:
            raise ValueError(
                "Only one active Nunchaku FLUX.2 Klein Ref Latent Weight "
                "specification is supported per MODEL branch."
            )
        if transformer_options.get(TEXT_REF_BALANCE_OPTION) is not None:
            raise ValueError(
                "Nunchaku FLUX.2 Klein Ref Latent Weight cannot be stacked "
                "with Text/Ref Balance in this release. Remove one node; "
                "prediction-residual composition is not defined."
            )
        if transformer_options.get(REF_LATENT_CONTROLLER_DIRECT_OPTION) is not None:
            raise ValueError(
                "Prediction-space Ref Latent Weight cannot be combined with "
                "Ref Latent Controller (Direct K/V). Remove one algorithm family."
            )

        branch = model.clone()
        branch.model_options["transformer_options"][REF_LATENT_WEIGHT_OPTION] = (
            KleinRefLatentWeightSpec(
                reference_index=reference_index,
                weight=float(weight),
            )
        )
        adapter.shared_lora_state.register_patcher(branch)
        return (branch,)
