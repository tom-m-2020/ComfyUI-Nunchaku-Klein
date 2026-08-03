"""Registration surface for target-owned Enhancer compatibility nodes."""

from .mask_reference import NunchakuKleinMaskReferenceController
from .multi_reference import NunchakuKleinMultiReferenceLatent


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinMaskReferenceController": NunchakuKleinMaskReferenceController,
    "NunchakuKleinMultiReferenceLatent": NunchakuKleinMultiReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinMaskReferenceController": (
        "Nunchaku FLUX.2 Klein Mask Ref Controller"
    ),
    "NunchakuKleinMultiReferenceLatent": (
        "Nunchaku FLUX.2 Klein Multi Reference Latent"
    ),
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NunchakuKleinMaskReferenceController",
    "NunchakuKleinMultiReferenceLatent",
]
