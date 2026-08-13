import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.supported_models
import folder_paths

from ..models.klein_wrapper import (
    LORA_SPEC_OPTION,
    NunchakuFlux2KleinAdapter,
    calculate_live_model_size,
)


_WRAPPER_KEY = "nunchaku_klein_force_full_load"
_CLONE_CALLBACK_KEY = "nunchaku_klein_shared_lora_state"


@dataclass(frozen=True)
class KleinArchitectureProfile:
    name: str
    context_dim: int
    num_heads: int
    num_double_blocks: int
    num_single_blocks: int


KLEIN_ARCHITECTURE_PROFILES = (
    KleinArchitectureProfile("4B", 7680, 24, 5, 20),
    KleinArchitectureProfile("9B", 12288, 32, 8, 24),
)


def _force_full_load(executor, model, noise_shape, conds, *args, **kwargs):
    kwargs["force_full_load"] = True
    return executor(model, noise_shape, conds, *args, **kwargs)


def _require_capabilities():
    try:
        from nunchaku.models.transformers.transformer_flux2 import (
            NunchakuFlux2Transformer2DModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Nunchaku FLUX.2 support is unavailable. Install a validated "
            "Nunchaku build containing "
            "nunchaku.models.transformers.transformer_flux2."
        ) from exc

    if not issubclass(NunchakuFlux2Transformer2DModel, torch.nn.Module):
        raise RuntimeError(
            "Unsupported Nunchaku FLUX.2 backend: transformer must inherit "
            "torch.nn.Module for ComfyUI-managed movement."
        )

    required_methods = ("from_pretrained", "forward")
    missing = [
        name
        for name in required_methods
        if not hasattr(NunchakuFlux2Transformer2DModel, name)
    ]
    if missing:
        raise RuntimeError(
            f"Unsupported Nunchaku FLUX.2 backend; missing methods {missing}."
        )

    pretrained_parameters = inspect.signature(
        NunchakuFlux2Transformer2DModel.from_pretrained
    ).parameters
    required_parameters = {
        "pretrained_model_name_or_path",
        "device",
        "torch_dtype",
        "offload",
        "return_metadata",
    }
    accepts_keyword_options = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in pretrained_parameters.values()
    )
    missing_parameters = sorted(required_parameters - pretrained_parameters.keys())
    if missing_parameters and not accepts_keyword_options:
        raise RuntimeError(
            "Unsupported Nunchaku from_pretrained signature; missing "
            f"parameters {missing_parameters}."
        )

    forward_parameters = inspect.signature(
        NunchakuFlux2Transformer2DModel.forward
    ).parameters
    required_forward_parameters = {
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
    }
    missing_forward_parameters = sorted(
        required_forward_parameters - forward_parameters.keys()
    )
    if missing_forward_parameters:
        raise RuntimeError(
            "Unsupported Nunchaku FLUX.2 forward signature; missing "
            f"parameters {missing_forward_parameters}."
        )

    prepare_parameters = inspect.signature(
        comfy.sampler_helpers.prepare_sampling
    ).parameters
    if "force_full_load" not in prepare_parameters:
        raise RuntimeError(
            "This loader requires ComfyUI prepare_sampling(force_full_load=...)."
        )
    if not hasattr(
        comfy.patcher_extension.WrappersMP,
        "PREPARE_SAMPLING",
    ) or not hasattr(comfy.model_patcher.ModelPatcher, "add_wrapper_with_key"):
        raise RuntimeError(
            "This loader requires ComfyUI's per-ModelPatcher "
            "PREPARE_SAMPLING wrapper API."
        )
    if "size" not in inspect.signature(
        comfy.model_patcher.ModelPatcher
    ).parameters:
        raise RuntimeError(
            "This loader requires ModelPatcher(..., size=...) support."
        )
    return NunchakuFlux2Transformer2DModel


def _parse_json_object(metadata: dict[str, str], key: str) -> dict:
    raw = metadata.get(key)
    if not raw:
        raise ValueError(f"Model metadata is missing required {key!r}.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model metadata {key!r} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata {key!r} must contain a JSON object.")
    return value


def _validated_comfy_config(
    metadata: dict[str, str],
) -> tuple[KleinArchitectureProfile, dict]:
    if metadata.get("model_class") != "NunchakuFlux2Transformer2DModel":
        raise ValueError(
            "Unsupported model metadata: model_class must be "
            "NunchakuFlux2Transformer2DModel."
        )

    config = _parse_json_object(metadata, "config")
    quantization = _parse_json_object(metadata, "quantization_config")

    shared_expected = {
        "_class_name": "Flux2Transformer2DModel",
        "in_channels": 128,
        "attention_head_dim": 128,
        "patch_size": 1,
        "axes_dims_rope": [32, 32, 32, 32],
        "rope_theta": 2000,
        "guidance_embeds": False,
        "eps": 1e-6,
        "mlp_ratio": 3.0,
        "timestep_guidance_channels": 256,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in shared_expected.items()
        if config.get(key) != value
    }
    if config.get("out_channels") not in (None, 128):
        mismatches["out_channels"] = {
            "expected": "null or 128",
            "actual": config.get("out_channels"),
        }

    expected_quantization = {
        "method": "svdquant",
        "rank": 32,
    }
    for key, value in expected_quantization.items():
        if quantization.get(key) != value:
            mismatches[f"quantization_config.{key}"] = {
                "expected": value,
                "actual": quantization.get(key),
            }
    for part in ("weight", "activation"):
        part_config = quantization.get(part)
        for key, value in {
            "dtype": "int4",
            "group_size": 64,
            "scale_dtype": None,
        }.items():
            if (
                not isinstance(part_config, dict)
                or key not in part_config
                or part_config[key] != value
            ):
                mismatches[f"quantization_config.{part}.{key}"] = {
                    "expected": value,
                    "actual": (
                        None if not isinstance(part_config, dict) else part_config.get(key)
                    ),
                }

    if mismatches:
        raise ValueError(
            "Unsupported FLUX.2 Klein metadata: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    matching_profiles = [
        profile
        for profile in KLEIN_ARCHITECTURE_PROFILES
        if config.get("joint_attention_dim") == profile.context_dim
        and config.get("num_attention_heads") == profile.num_heads
        and config.get("num_layers") == profile.num_double_blocks
        and config.get("num_single_layers") == profile.num_single_blocks
    ]
    if len(matching_profiles) != 1:
        actual = {
            "joint_attention_dim": config.get("joint_attention_dim"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_layers": config.get("num_layers"),
            "num_single_layers": config.get("num_single_layers"),
        }
        raise ValueError(
            "Unsupported FLUX.2 Klein architecture profile: "
            f"{json.dumps(actual, sort_keys=True)}. Expected exactly one of "
            f"{[profile.name for profile in KLEIN_ARCHITECTURE_PROFILES]}."
        )
    profile = matching_profiles[0]

    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    comfy_config = {
        "image_model": "flux2",
        "in_channels": config["in_channels"],
        "out_channels": config["out_channels"] or config["in_channels"],
        "hidden_size": hidden_size,
        "context_in_dim": config["joint_attention_dim"],
        "num_heads": config["num_attention_heads"],
        "depth": config["num_layers"],
        "depth_single_blocks": config["num_single_layers"],
        "axes_dim": list(config["axes_dims_rope"]),
        "theta": config["rope_theta"],
        "patch_size": config["patch_size"],
        "guidance_embed": config["guidance_embeds"],
        "disable_unet_model_creation": True,
    }
    return profile, comfy_config


def _read_metadata(model_path: Path) -> dict[str, str]:
    try:
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception as exc:
        raise ValueError(
            f"Unable to read safetensors metadata from {model_path}."
        ) from exc
    if not metadata:
        raise ValueError(f"Model {model_path} has no safetensors metadata.")
    return metadata


class NunchakuKleinModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def load_model(self, model_name: str):
        transformer_class = _require_capabilities()
        model_path = Path(
            folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        )
        metadata = _read_metadata(model_path)
        profile, comfy_config = _validated_comfy_config(metadata)

        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        if not isinstance(load_device, torch.device) or load_device.type != "cuda":
            raise RuntimeError(
                "The validated FLUX.2 Klein path requires a CUDA load device, "
                f"got {load_device!r}."
            )
        if not isinstance(offload_device, torch.device):
            raise RuntimeError(
                "ComfyUI unet_offload_device() must return a torch.device, got "
                f"{offload_device!r}."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"Device {load_device} does not report bfloat16 support."
            )

        dtype = torch.bfloat16
        transformer = None
        patcher = None
        try:
            transformer, returned_metadata = transformer_class.from_pretrained(
                model_path,
                device=load_device,
                torch_dtype=dtype,
                offload=False,
                return_metadata=True,
            )
            returned_profile, returned_config = _validated_comfy_config(
                returned_metadata
            )
            if returned_profile != profile or returned_config != comfy_config:
                raise RuntimeError(
                    "Nunchaku returned metadata that differs from the "
                    "validated safetensors header."
                )

            adapter = NunchakuFlux2KleinAdapter(
                transformer,
                in_channels=comfy_config["in_channels"],
                context_dim=comfy_config["context_in_dim"],
                patch_size=comfy_config["patch_size"],
                axes_dim=tuple(comfy_config["axes_dim"]),
                dtype=dtype,
                architecture_profile=profile.name,
            )

            model_config = comfy.supported_models.Flux2(comfy_config)
            model_config.set_inference_dtype(dtype, None)
            base_model = model_config.get_model({})
            if getattr(base_model, "diffusion_model", None) is not None:
                raise RuntimeError(
                    "ComfyUI created an unexpected full-precision Flux2 "
                    "transformer despite disable_unet_model_creation."
                )
            base_model.diffusion_model = adapter

            live_size = calculate_live_model_size(base_model, transformer)
            patcher = comfy.model_patcher.ModelPatcher(
                base_model,
                load_device,
                offload_device,
                size=live_size,
            )
            patcher.model_options["transformer_options"][LORA_SPEC_OPTION] = ()
            patcher.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                _WRAPPER_KEY,
                _force_full_load,
            )
            adapter.shared_lora_state.current_size = live_size
            adapter.shared_lora_state.register_patcher(patcher)

            def register_clone(_source, clone):
                adapter.shared_lora_state.register_patcher(clone)

            patcher.add_callback_with_key(
                comfy.patcher_extension.CallbacksMP.ON_CLONE,
                _CLONE_CALLBACK_KEY,
                register_clone,
            )

            # from_pretrained constructs on CUDA. Forced registration records
            # that residency without an unnecessary CUDA -> CPU -> CUDA trip.
            model_management.load_models_gpu([patcher], force_full_load=True)
            if patcher.loaded_size() != live_size:
                raise RuntimeError(
                    "ComfyUI did not register the complete Nunchaku model: "
                    f"loaded {patcher.loaded_size()} of {live_size} bytes."
                )
            return (patcher,)
        except Exception:
            if patcher is not None:
                model_management.unload_model_and_clones(
                    patcher,
                    all_devices=True,
                )
            elif transformer is not None:
                transformer.to(device=offload_device)
            model_management.cleanup_models_gc()
            model_management.soft_empty_cache()
            raise
