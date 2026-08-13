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
from .klein_lora_normalization import normalize_klein_lora_state


def _get_inherited_lora_specs(model) -> tuple[KleinLoraSpec, ...]:
    transformer_options = model.model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        raise RuntimeError(
            "Nunchaku Klein MODEL has invalid transformer options."
        )
    inherited = transformer_options.get(LORA_SPEC_OPTION, ())
    if not isinstance(inherited, tuple) or not all(
        isinstance(spec, KleinLoraSpec) for spec in inherited
    ):
        raise RuntimeError(
            f"{LORA_SPEC_OPTION} must contain a tuple of KleinLoraSpec."
        )
    return inherited


def _validate_lora_profile(metadata, profile: str, lora_path: Path) -> None:
    base_model = (metadata or {}).get("ss_base_model_version")
    if base_model is None:
        return
    normalized = base_model.lower().replace("-", "_")
    expected = ("flux2", "klein", profile.lower())
    if not all(part in normalized for part in expected):
        raise ValueError(
            f"LoRA {lora_path} targets {base_model!r}, not "
            f"FLUX.2 Klein {profile}."
        )


def _lora_cache_identity(lora_path: Path, stat, profile: str) -> tuple:
    return (str(lora_path), stat.st_size, stat.st_mtime_ns, profile)


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
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a compatible FLUX.2 Klein LoRA as branch-local execution state."

    def load_lora(self, model, lora_name: str, strength: float):
        if not math.isfinite(strength):
            raise ValueError(f"LoRA strength must be finite, got {strength}.")
        adapter = getattr(model.model, "diffusion_model", None)
        if not isinstance(adapter, NunchakuFlux2KleinAdapter):
            raise TypeError(
                "Nunchaku Klein LoRA loading requires a MODEL from "
                "NunchakuKleinModelLoader."
            )
        profile = adapter.architecture_profile
        inherited = _get_inherited_lora_specs(model)

        if strength == 0.0:
            branch = model.clone()
            branch.model_options["transformer_options"][LORA_SPEC_OPTION] = inherited
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
        fingerprint = _lora_cache_identity(lora_path, stat, profile)
        if self.loaded_lora is not None and self.loaded_lora[0] == fingerprint:
            state_dict, metadata = self.loaded_lora[1:]
        else:
            raw_state_dict, metadata = comfy.utils.load_torch_file(
                str(lora_path),
                safe_load=True,
                return_metadata=True,
            )
            if not raw_state_dict:
                raise ValueError(f"LoRA {lora_path} contains no tensors.")
            if not all(torch.is_tensor(tensor) for tensor in raw_state_dict.values()):
                raise ValueError(f"LoRA {lora_path} contains non-tensor weights.")

            _validate_lora_profile(metadata, profile, lora_path)

            try:
                state_dict = normalize_klein_lora_state(raw_state_dict)
            except (TypeError, ValueError) as error:
                raise ValueError(f"LoRA {lora_path} is not supported: {error}") from error

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
                profile=profile,
            )
            self._validate_pairs(
                transformer,
                lora_path,
                unquantized,
                suffixes=(".lora_A.weight", ".lora_B.weight"),
                profile=profile,
            )
            self.loaded_lora = (fingerprint, state_dict, metadata)

        branch = model.clone()
        branch.model_options["transformer_options"][LORA_SPEC_OPTION] = (
            *inherited,
            KleinLoraSpec(
                path=lora_path.resolve(),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                strength=float(strength),
                state_dict=state_dict,
            ),
        )
        adapter.shared_lora_state.register_patcher(branch)
        return (branch,)

    @staticmethod
    def _validate_pairs(transformer, lora_path, weights, *, suffixes, profile):
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
                    f"LoRA pair {prefix!r} is incompatible with Klein {profile}: "
                    f"got {list(first.shape)} and {list(second.shape)}, "
                    f"expected input/output dimensions "
                    f"{module.in_features}/{module.out_features}."
                )
