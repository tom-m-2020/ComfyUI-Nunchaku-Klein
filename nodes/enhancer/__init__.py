"""Registration surface for target-owned Enhancer compatibility nodes."""

from .klein_enhancer import NunchakuKleinEnhancer
from .mask_reference import NunchakuKleinMaskReferenceController
from .multi_reference import NunchakuKleinMultiReferenceLatent
from .text_enhancer import NunchakuKleinTextEnhancer


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinEnhancer": NunchakuKleinEnhancer,
    "NunchakuKleinMaskReferenceController": NunchakuKleinMaskReferenceController,
    "NunchakuKleinMultiReferenceLatent": NunchakuKleinMultiReferenceLatent,
    "NunchakuKleinTextEnhancer": NunchakuKleinTextEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinEnhancer": "Nunchaku FLUX.2 Klein Enhancer",
    "NunchakuKleinMaskReferenceController": (
        "Nunchaku FLUX.2 Klein Mask Ref Controller"
    ),
    "NunchakuKleinMultiReferenceLatent": (
        "Nunchaku FLUX.2 Klein Multi Reference Latent"
    ),
    "NunchakuKleinTextEnhancer": "Nunchaku FLUX.2 Klein Text Enhancer",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NunchakuKleinEnhancer",
    "NunchakuKleinMaskReferenceController",
    "NunchakuKleinMultiReferenceLatent",
    "NunchakuKleinTextEnhancer",
]
