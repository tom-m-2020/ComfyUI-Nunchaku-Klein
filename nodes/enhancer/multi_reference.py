"""Target-owned ordered multi-reference conditioning node.

Behavior is adapted from ComfyUI-Flux2Klein-Enhancer under the MIT License;
see THIRD_PARTY_NOTICES.md.
"""

from .common import REFERENCE_CATEGORY
from .conditioning import clone_conditioning
from .metadata import set_reference_latents, split_latent_batch


class NunchakuKleinMultiReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent_1": ("LATENT",),
            },
            "optional": {
                f"latent_{index}": ("LATENT",) for index in range(2, 9)
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = REFERENCE_CATEGORY
    DESCRIPTION = "Adds up to eight ordered FLUX.2 reference latent inputs."

    def apply(self, conditioning, latent_1, **optional):
        inputs = {"latent_1": latent_1, **optional}
        references = []
        for index in range(1, 9):
            name = f"latent_{index}"
            if name in inputs and inputs[name] is not None:
                references.extend(split_latent_batch(inputs[name], name=name))

        output = clone_conditioning(conditioning)
        for item in output:
            item[1] = set_reference_latents(item[1], references, method="index")
        return (output,)


__all__ = ["NunchakuKleinMultiReferenceLatent"]
