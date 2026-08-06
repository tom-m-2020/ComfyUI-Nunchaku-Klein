"""Registration surface for target-owned Enhancer compatibility nodes."""

from .klein_enhancer import NunchakuKleinEnhancer
from .detail_controller import NunchakuKleinDetailController
from .mask_reference import NunchakuKleinMaskReferenceController
from .multi_reference import NunchakuKleinMultiReferenceLatent
from .sectioned_encoder import NunchakuKleinSectionedEncoder
from .text_enhancer import NunchakuKleinTextEnhancer


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinEnhancer": NunchakuKleinEnhancer,
    "NunchakuKleinDetailController": NunchakuKleinDetailController,
    "NunchakuKleinMaskReferenceController": NunchakuKleinMaskReferenceController,
    "NunchakuKleinMultiReferenceLatent": NunchakuKleinMultiReferenceLatent,
    "NunchakuKleinSectionedEncoder": NunchakuKleinSectionedEncoder,
    "NunchakuKleinTextEnhancer": NunchakuKleinTextEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinEnhancer": "Nunchaku FLUX.2 Klein Enhancer",
    "NunchakuKleinDetailController": "Nunchaku FLUX.2 Klein Detail Controller",
    "NunchakuKleinMaskReferenceController": (
        "Nunchaku FLUX.2 Klein Mask Ref Controller"
    ),
    "NunchakuKleinMultiReferenceLatent": (
        "Nunchaku FLUX.2 Klein Multi Reference Latent"
    ),
    "NunchakuKleinSectionedEncoder": (
        "Nunchaku FLUX.2 Klein Sectioned Encoder"
    ),
    "NunchakuKleinTextEnhancer": "Nunchaku FLUX.2 Klein Text Enhancer",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NunchakuKleinEnhancer",
    "NunchakuKleinDetailController",
    "NunchakuKleinMaskReferenceController",
    "NunchakuKleinMultiReferenceLatent",
    "NunchakuKleinSectionedEncoder",
    "NunchakuKleinTextEnhancer",
]
