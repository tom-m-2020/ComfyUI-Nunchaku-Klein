"""Registration surface for target-owned Enhancer compatibility nodes."""

from .color_anchor import NunchakuKleinColorAnchor
from .klein_enhancer import NunchakuKleinEnhancer
from .detail_controller import NunchakuKleinDetailController
from .identity_guidance import NunchakuKleinIdentityGuidance
from .mask_reference import NunchakuKleinMaskReferenceController
from .multi_reference import NunchakuKleinMultiReferenceLatent
from .ref_latent_weight import NunchakuKleinRefLatentWeight
from .ref_latent_weight_direct import NunchakuKleinRefLatentWeightDirectKV
from .sectioned_encoder import NunchakuKleinSectionedEncoder
from .text_enhancer import NunchakuKleinTextEnhancer
from .text_ref_balance import NunchakuKleinTextRefBalance


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinColorAnchor": NunchakuKleinColorAnchor,
    "NunchakuKleinEnhancer": NunchakuKleinEnhancer,
    "NunchakuKleinDetailController": NunchakuKleinDetailController,
    "NunchakuKleinIdentityGuidance": NunchakuKleinIdentityGuidance,
    "NunchakuKleinMaskReferenceController": NunchakuKleinMaskReferenceController,
    "NunchakuKleinMultiReferenceLatent": NunchakuKleinMultiReferenceLatent,
    "NunchakuKleinRefLatentWeight": NunchakuKleinRefLatentWeight,
    "NunchakuKleinRefLatentWeightDirectKV": NunchakuKleinRefLatentWeightDirectKV,
    "NunchakuKleinSectionedEncoder": NunchakuKleinSectionedEncoder,
    "NunchakuKleinTextEnhancer": NunchakuKleinTextEnhancer,
    "NunchakuKleinTextRefBalance": NunchakuKleinTextRefBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinColorAnchor": "Nunchaku FLUX.2 Klein Color Anchor",
    "NunchakuKleinEnhancer": "Nunchaku FLUX.2 Klein Enhancer",
    "NunchakuKleinDetailController": "Nunchaku FLUX.2 Klein Detail Controller",
    "NunchakuKleinIdentityGuidance": "Nunchaku FLUX.2 Klein Identity Guidance",
    "NunchakuKleinMaskReferenceController": (
        "Nunchaku FLUX.2 Klein Mask Ref Controller"
    ),
    "NunchakuKleinMultiReferenceLatent": (
        "Nunchaku FLUX.2 Klein Multi Reference Latent"
    ),
    "NunchakuKleinRefLatentWeight": (
        "Nunchaku FLUX.2 Klein Ref Latent Weight"
    ),
    "NunchakuKleinRefLatentWeightDirectKV": (
        "Nunchaku FLUX.2 Klein Ref Latent Weight (Direct K/V)"
    ),
    "NunchakuKleinSectionedEncoder": (
        "Nunchaku FLUX.2 Klein Sectioned Encoder"
    ),
    "NunchakuKleinTextEnhancer": "Nunchaku FLUX.2 Klein Text Enhancer",
    "NunchakuKleinTextRefBalance": "Nunchaku FLUX.2 Klein Text/Ref Balance",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NunchakuKleinColorAnchor",
    "NunchakuKleinEnhancer",
    "NunchakuKleinDetailController",
    "NunchakuKleinIdentityGuidance",
    "NunchakuKleinMaskReferenceController",
    "NunchakuKleinMultiReferenceLatent",
    "NunchakuKleinRefLatentWeight",
    "NunchakuKleinRefLatentWeightDirectKV",
    "NunchakuKleinSectionedEncoder",
    "NunchakuKleinTextEnhancer",
    "NunchakuKleinTextRefBalance",
]
