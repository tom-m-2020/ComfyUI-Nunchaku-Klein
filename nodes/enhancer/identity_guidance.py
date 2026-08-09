"""Nunchaku-compatible FLUX.2 Klein Identity Guidance.

Behavior and UI are adapted from ComfyUI-Flux2Klein-Enhancer under the MIT
License; see THIRD_PARTY_NOTICES.md.
"""

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .common import POST_CFG_CATEGORY
from .post_cfg import register_post_cfg_callback
from .validation import validate_bchw_tensor, validate_finite_range


class NunchakuKleinIdentityGuidance:
    """Correct the denoised prediction through ComfyUI's post-CFG boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "identity_latent": ("LATENT", {
                    "tooltip": "VAE-encoded reference image at full resolution.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How hard to pull toward the reference each step. 0.3 = move 30% of the distance.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When to start correcting. 0.0 = beginning of denoising.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When to stop correcting. 0.8 = last 20% runs freely for texture refinement.",
                }),
                "mode": (["adaptive", "direct", "channel_match"], {
                    "default": "adaptive",
                    "tooltip": "adaptive: pulls only where prediction resembles reference. direct: pulls everywhere equally. channel_match: matches color/feature statistics without copying spatial content.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = POST_CFG_CATEGORY

    def apply(
        self,
        model,
        identity_latent,
        strength=0.3,
        start_percent=0.0,
        end_percent=0.8,
        mode="adaptive",
    ):
        strength = validate_finite_range(
            strength, name="strength", minimum=0.0, maximum=1.0
        )
        start_percent = validate_finite_range(
            start_percent, name="start_percent", minimum=0.0, maximum=1.0
        )
        end_percent = validate_finite_range(
            end_percent, name="end_percent", minimum=0.0, maximum=1.0
        )
        if mode not in ("adaptive", "direct", "channel_match"):
            raise ValueError(
                "mode must be 'adaptive', 'direct', or 'channel_match', "
                f"got {mode!r}."
            )
        if strength == 0.0:
            return (model,)
        if not isinstance(identity_latent, Mapping):
            raise TypeError("identity_latent must be a LATENT mapping.")
        reference = validate_bchw_tensor(
            identity_latent.get("samples"), name="identity_latent['samples']"
        )
        if not torch.isfinite(reference).all().item():
            raise ValueError("identity_latent['samples'] must contain only finite values.")

        # The callback needs the full latent, but must not retain a workflow CUDA
        # tensor or alias mutable LATENT storage across cached executions.
        reference = reference.detach().to(device="cpu", copy=True)

        def identity_guidance(args):
            if not isinstance(args, Mapping):
                raise TypeError("Identity Guidance post-CFG arguments must be a mapping.")
            denoised = args.get("denoised")
            sigma = args.get("sigma")
            if not torch.is_tensor(denoised) or denoised.ndim != 4:
                shape = getattr(denoised, "shape", None)
                raise ValueError(
                    "Identity Guidance requires BCHW args['denoised']; "
                    f"got {shape}."
                )
            if not torch.is_tensor(sigma) or sigma.numel() == 0:
                raise TypeError(
                    "Identity Guidance requires a non-empty tensor args['sigma']."
                )

            progress = max(0.0, min(1.0, 1.0 - float(sigma.flatten()[0])))
            if progress < start_percent or progress > end_percent:
                return denoised

            ref_resized = reference.to(
                device=denoised.device, dtype=denoised.dtype
            )
            if ref_resized.shape[0] != denoised.shape[0]:
                ref_resized = ref_resized[:1].expand(
                    denoised.shape[0], -1, -1, -1
                )
            if ref_resized.shape[2:] != denoised.shape[2:]:
                ref_resized = F.interpolate(
                    ref_resized,
                    size=denoised.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if ref_resized.shape[1] != denoised.shape[1]:
                if ref_resized.shape[1] > denoised.shape[1]:
                    ref_resized = ref_resized[:, : denoised.shape[1]]
                else:
                    ref_resized = F.pad(
                        ref_resized,
                        (0, 0, 0, 0, 0, denoised.shape[1] - ref_resized.shape[1]),
                    )

            if mode == "direct":
                return denoised + (ref_resized - denoised) * strength
            if mode == "adaptive":
                similarity = F.cosine_similarity(
                    denoised.flatten(2), ref_resized.flatten(2), dim=1
                )
                weight = similarity.clamp(0.0, 1.0).unsqueeze(1).view(
                    denoised.shape[0], 1, denoised.shape[2], denoised.shape[3]
                )
                return denoised + (ref_resized - denoised) * weight * strength

            ref_mean = ref_resized.mean(dim=(2, 3), keepdim=True)
            ref_std = ref_resized.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
            den_mean = denoised.mean(dim=(2, 3), keepdim=True)
            den_std = denoised.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
            matched = (denoised - den_mean) / den_std * ref_std + ref_mean
            return denoised + (matched - denoised) * strength

        return (register_post_cfg_callback(model, identity_guidance),)


__all__ = ["NunchakuKleinIdentityGuidance"]
