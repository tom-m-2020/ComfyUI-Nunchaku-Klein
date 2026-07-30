from .nodes.klein_loader import NunchakuKleinModelLoader


NODE_CLASS_MAPPINGS = {
    "NunchakuKleinModelLoader": NunchakuKleinModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuKleinModelLoader": "Nunchaku FLUX.2 Klein Model Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
