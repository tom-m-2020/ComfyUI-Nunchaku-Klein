"""Direct pre-attention spatial K/V control for one FLUX.2 Klein reference."""

from dataclasses import dataclass
import logging

import torch

from ...models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_CONTROLLER_DIRECT_OPTION,
    REF_LATENT_WEIGHT_OPTION,
    TEXT_REF_BALANCE_OPTION,
)
from .attention_kv import validate_attention_kv_layout
from .common import REFERENCE_CATEGORY
from .validation import (
    validate_bool,
    validate_conditioning,
    validate_finite_range,
    validate_int_range,
)


logger = logging.getLogger(__name__)
SPATIAL_FADE_MODES = ("none", "center_out", "edges_out", "top_down", "left_right")


@dataclass(frozen=True)
class KleinRefLatentControllerKVCallback:
    strength: float
    reference_index: int
    spatial_fade: str
    spatial_fade_strength: float
    debug: bool = False

    def __call__(self, query, key, value, metadata):
        _, _, reference_ranges = validate_attention_kv_layout(
            query, key, value, metadata, name="Direct K/V Ref Latent Controller"
        )
        shapes = getattr(metadata, "reference_spatial_shapes", None)
        if not isinstance(shapes, tuple) or len(shapes) != len(reference_ranges):
            raise ValueError(
                "Direct K/V Ref Latent Controller requires one immutable "
                "reference_spatial_shapes entry per runtime reference."
            )
        for index, (shape, token_range) in enumerate(zip(shapes, reference_ranges, strict=True)):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
            ):
                raise ValueError(f"reference_spatial_shapes[{index}] must be (height, width).")
            if shape[0] * shape[1] != token_range[1] - token_range[0]:
                raise ValueError(
                    f"reference_spatial_shapes[{index}] does not match its token count."
                )
        if not reference_ranges:
            return None
        if not 0 <= self.reference_index < len(reference_ranges):
            raise IndexError(
                "Direct K/V Ref Latent Controller reference_index "
                f"{self.reference_index} is out of range for "
                f"{len(reference_ranges)} runtime references."
            )

        start, end = reference_ranges[self.reference_index]
        if self.strength == 1.0 and (
            self.spatial_fade == "none" or self.spatial_fade_strength == 0.0
        ):
            return None
        scale = self.strength
        if self.spatial_fade != "none":
            height, width = shapes[self.reference_index]
            y = torch.linspace(0.0, 1.0, height, device=key.device)
            x = torch.linspace(0.0, 1.0, width, device=key.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            if self.spatial_fade in ("center_out", "edges_out"):
                distance = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
                distance = distance / distance.max().clamp(min=1e-8)
                if self.spatial_fade == "center_out":
                    weights = 1.0 - distance * self.spatial_fade_strength
                else:
                    weights = 1.0 - self.spatial_fade_strength + distance * self.spatial_fade_strength
            elif self.spatial_fade == "top_down":
                weights = 1.0 - yy * self.spatial_fade_strength
            else:
                weights = 1.0 - xx * self.spatial_fade_strength
            scale = (self.strength * weights.flatten()).to(dtype=key.dtype).view(1, 1, -1, 1)

        if not (isinstance(scale, float) and scale == 1.0):
            key[:, :, start:end, :].mul_(scale)
            value[:, :, start:end, :].mul_(scale)
        if self.debug:
            logger.info(
                "Ref Latent Controller Direct K/V: %s %d ref=%d range=[%d,%d) "
                "strength=%.3f fade=%s",
                metadata.block_type,
                metadata.block_index,
                self.reference_index,
                start,
                end,
                self.strength,
                self.spatial_fade,
            )
        return None


class NunchakuKleinRefLatentControllerDirectKV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 7}),
            },
            "optional": {
                "spatial_fade": (list(SPATIAL_FADE_MODES), {"default": "none"}),
                "spatial_fade_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = (
        "Directly scales one reference's projected attention K/V tokens, "
        "optionally with a spatial fade."
    )

    def control(
        self,
        model,
        conditioning,
        strength=1.0,
        reference_index=0,
        spatial_fade="none",
        spatial_fade_strength=0.5,
        debug=False,
    ):
        validate_conditioning(conditioning)
        strength = validate_finite_range(strength, name="strength", minimum=0.0, maximum=1000.0)
        reference_index = validate_int_range(reference_index, name="reference_index", minimum=0, maximum=7)
        if spatial_fade not in SPATIAL_FADE_MODES:
            raise ValueError(f"spatial_fade must be one of {SPATIAL_FADE_MODES}, got {spatial_fade!r}.")
        spatial_fade_strength = validate_finite_range(
            spatial_fade_strength, name="spatial_fade_strength", minimum=0.0, maximum=1.0
        )
        debug = validate_bool(debug, name="debug")

        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku FLUX.2 Klein Ref Latent Controller (Direct K/V) "
                "requires a MODEL from NunchakuKleinModelLoader."
            )
        options = model.model_options.get("transformer_options")
        if not isinstance(options, dict):
            raise RuntimeError("Nunchaku Klein MODEL has invalid transformer options.")
        if options.get(REF_LATENT_WEIGHT_OPTION) is not None or options.get(TEXT_REF_BALANCE_OPTION) is not None:
            raise ValueError(
                "Nunchaku FLUX.2 Klein Ref Latent Controller (Direct K/V) cannot "
                "be combined with prediction-space Ref Latent Weight or Text/Ref Balance."
            )
        callbacks = options.get(ATTENTION_CALLBACKS_OPTION, Flux2AttentionCallbacks())
        if not isinstance(callbacks, Flux2AttentionCallbacks):
            raise TypeError(f"{ATTENTION_CALLBACKS_OPTION} must contain Flux2AttentionCallbacks.")
        inherited = options.get(REF_LATENT_CONTROLLER_DIRECT_OPTION, ())
        if not isinstance(inherited, tuple) or not all(callable(item) for item in inherited):
            raise TypeError(f"{REF_LATENT_CONTROLLER_DIRECT_OPTION} must contain an immutable tuple of callables.")

        callback = KleinRefLatentControllerKVCallback(
            float(strength), reference_index, spatial_fade, float(spatial_fade_strength), debug
        )
        branch = model.clone()
        branch_options = branch.model_options["transformer_options"]
        branch_options[ATTENTION_CALLBACKS_OPTION] = Flux2AttentionCallbacks(
            (*callbacks.pre_attention_callbacks, callback), callbacks.post_attention_callbacks
        )
        branch_options[REF_LATENT_CONTROLLER_DIRECT_OPTION] = (*inherited, callback)
        adapter.shared_lora_state.register_patcher(branch)
        return branch, conditioning


__all__ = ["KleinRefLatentControllerKVCallback", "NunchakuKleinRefLatentControllerDirectKV"]
