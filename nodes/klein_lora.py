from pathlib import Path
import math

import torch

import comfy.utils
import folder_paths

from ..models.klein_wrapper import (
    KleinLoraSpec,
    LORA_SPEC_OPTION,
    NunchakuFlux2KleinAdapter,
)


class NunchakuKleinLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a FLUX.2 Klein 9B LoRA as branch-local execution state."
    )

    def load_lora(self, model, lora_name: str, strength: float):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError(
                f"LoRA strength must be a finite non-negative value, got {strength}."
            )
        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku Klein LoRA loading requires a MODEL from "
                "NunchakuKleinModelLoader."
            )

        if strength == 0.0:
            branch = model.clone()
            branch.model_options["transformer_options"].pop(
                LORA_SPEC_OPTION,
                None,
            )
            return (branch,)

        transformer = adapter.transformer
        adapter.shared_lora_state.require_lora_capabilities()
        convert_lora = getattr(transformer, "_convert_lora_keys", None)
        if not callable(convert_lora):
            raise RuntimeError(
                "The installed Nunchaku FLUX.2 backend does not expose its "
                "LoRA conversion parser."
            )

        lora_path = Path(
            folder_paths.get_full_path_or_raise("loras", lora_name)
        )
        stat = lora_path.stat()
        fingerprint = (str(lora_path), stat.st_size, stat.st_mtime_ns)
        if self.loaded_lora is not None and self.loaded_lora[0] == fingerprint:
            state_dict, metadata = self.loaded_lora[1:]
        else:
            state_dict, metadata = comfy.utils.load_torch_file(
                str(lora_path),
                safe_load=True,
                return_metadata=True,
            )
            if not state_dict:
                raise ValueError(f"LoRA {lora_path} contains no tensors.")
            if not all(torch.is_tensor(tensor) for tensor in state_dict.values()):
                raise ValueError(f"LoRA {lora_path} contains non-tensor weights.")

            base_model = (metadata or {}).get("ss_base_model_version")
            if base_model is not None:
                normalized = base_model.lower().replace("-", "_")
                if not all(part in normalized for part in ("flux2", "klein", "9b")):
                    raise ValueError(
                        f"LoRA {lora_path} targets {base_model!r}, not "
                        "FLUX.2 Klein 9B."
                    )

            # Vitoom's parser is version-pinned private API, but it is the only
            # backend path that normalizes every supported Klein LoRA key format.
            quantized, unquantized = convert_lora(state_dict)
            if not quantized and not unquantized:
                raise ValueError(
                    f"LoRA {lora_path} has no weights recognized by the installed "
                    "Nunchaku FLUX.2 Klein parser."
                )

            self._validate_pairs(
                transformer,
                lora_path,
                quantized,
                suffixes=(".proj_down", ".proj_up"),
            )
            self._validate_pairs(
                transformer,
                lora_path,
                unquantized,
                suffixes=(".lora_A.weight", ".lora_B.weight"),
            )
            self.loaded_lora = (fingerprint, state_dict, metadata)

        branch = model.clone()
        branch.model_options["transformer_options"][LORA_SPEC_OPTION] = (
            KleinLoraSpec(
                path=lora_path.resolve(),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                strength=float(strength),
                state_dict=state_dict,
            )
        )
        adapter.shared_lora_state.register_patcher(branch)
        return (branch,)

    @staticmethod
    def _validate_pairs(transformer, lora_path, weights, *, suffixes):
        first_suffix, second_suffix = suffixes
        first_keys = sorted(
            key for key in weights if key.endswith(first_suffix)
        )
        if len(weights) != len(first_keys) * 2:
            raise ValueError(
                f"LoRA {lora_path} produced incomplete Nunchaku weight pairs."
            )

        for first_key in first_keys:
            prefix = first_key[: -len(first_suffix)]
            second_key = prefix + second_suffix
            if second_key not in weights:
                raise ValueError(
                    f"LoRA {lora_path} is missing the pair for {first_key!r}."
                )

            first = weights[first_key]
            second = weights[second_key]
            if first.ndim != 2 or second.ndim != 2:
                raise ValueError(
                    f"LoRA pair {prefix!r} must contain two matrices, got "
                    f"{list(first.shape)} and {list(second.shape)}."
                )
            if first.shape[0] != second.shape[1]:
                raise ValueError(
                    f"LoRA pair {prefix!r} has inconsistent rank: "
                    f"{list(first.shape)} and {list(second.shape)}."
                )

            module = transformer.get_submodule(prefix)
            if (
                first.shape[1] != module.in_features
                or second.shape[0] != module.out_features
            ):
                raise ValueError(
                    f"LoRA pair {prefix!r} is incompatible with Klein 9B: "
                    f"got {list(first.shape)} and {list(second.shape)}, "
                    f"expected input/output dimensions "
                    f"{module.in_features}/{module.out_features}."
                )
