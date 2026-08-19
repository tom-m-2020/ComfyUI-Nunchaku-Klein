from .nodes.klein_loader import NunchakuKleinModelLoader
from .nodes.klein_lora import (
    NunchakuKleinLoraLoader,
    NunchakuKleinPowerLoraLoader,
)
from .nodes.enhancer import (
    NODE_CLASS_MAPPINGS as ENHANCER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ENHANCER_NODE_DISPLAY_NAME_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {
    **ENHANCER_NODE_CLASS_MAPPINGS,
    "NunchakuKleinLoraLoader": NunchakuKleinLoraLoader,
    "NunchakuKleinPowerLoraLoader": NunchakuKleinPowerLoraLoader,
    "NunchakuKleinModelLoader": NunchakuKleinModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ENHANCER_NODE_DISPLAY_NAME_MAPPINGS,
    "NunchakuKleinLoraLoader": (
        "Nunchaku FLUX.2 Klein LoRA Loader"
    ),
    "NunchakuKleinPowerLoraLoader": (
        "Nunchaku FLUX.2 Klein Power LoRA Loader"
    ),
    "NunchakuKleinModelLoader": "Nunchaku FLUX.2 Klein Model Loader",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
