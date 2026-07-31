from .nodes.klein_loader import NunchakuKleinModelLoader
from .nodes.klein_lora import NunchakuKleinLoraLoader


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinLoraLoader": NunchakuKleinLoraLoader,
    "NunchakuKleinModelLoader": NunchakuKleinModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinLoraLoader": (
        "Nunchaku FLUX.2 Klein LoRA Loader"
    ),
    "NunchakuKleinModelLoader": "Nunchaku FLUX.2 Klein Model Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
