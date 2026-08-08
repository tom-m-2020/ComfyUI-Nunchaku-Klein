from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import threading
import weakref

import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

LORA_SPEC_OPTION = "nunchaku_klein_lora_spec"
REF_LATENT_WEIGHT_OPTION = "nunchaku_klein_ref_latent_weight"
_SHARED_LORA_STATE_ATTRIBUTE = "_comfyui_nunchaku_klein_lora_state"
# ComfyUI 0.29's Flux2 detection and Diffusers' Klein pipeline both separate
# reference images on RoPE axis 0 with indices 10, 20, ... .
REFERENCE_IMAGE_INDEX_STRIDE = 10.0

NUNCHAKU_TENSOR_DICTIONARIES = (
    "_quantized_part_sd",
    "_unquantized_part_sd",
    "_cached_quantized_loras",
    "_cached_unquantized_loras",
)


@dataclass(frozen=True)
class KleinLoraSpec:
    path: Path
    size: int
    mtime_ns: int
    strength: float
    state_dict: dict[str, torch.Tensor]

    @property
    def source_identity(self) -> tuple:
        return (
            "lora",
            str(self.path),
            self.size,
            self.mtime_ns,
        )

    @property
    def identity(self) -> tuple:
        return (
            *self.source_identity,
            self.strength,
        )


@dataclass(frozen=True)
class KleinRefLatentWeightSpec:
    """Prediction-residual substitute for one unreachable reference K/V weight."""

    reference_index: int
    weight: float


class KleinLoraPreparationCache:
    """Bounded cache of immutable states composed from complete flat tuples."""

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 2:
            raise ValueError("The prepared LoRA cache requires at least two entries.")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[tuple, ...], dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self._active_identity: tuple[tuple, ...] | None = None

    def prepare(
        self,
        desired: tuple[KleinLoraSpec, ...],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(desired, tuple):
            raise TypeError("Prepared Klein LoRAs require an ordered tuple.")
        if not desired:
            raise ValueError("Cannot compose an empty LoRA collection.")
        for spec in desired:
            if not isinstance(spec, KleinLoraSpec):
                raise TypeError("Prepared Klein LoRAs require KleinLoraSpec entries.")
            if not math.isfinite(spec.strength):
                raise ValueError(
                    f"LoRA strength must be finite, got {spec.strength} for {spec.path}."
                )

        identity = tuple(spec.identity for spec in desired)
        cached = self._entries.get(identity)
        if cached is not None:
            self._entries.move_to_end(identity)
            logger.info(
                "Prepared LoRA cache: HIT %s (%d entries, %.2f MiB)",
                " + ".join(spec.path.name for spec in desired),
                len(self._entries),
                self.unique_tensor_bytes() / (1024 * 1024),
            )
            return cached

        logger.info(
            "Prepared LoRA cache: MISS %s (%d entries, %.2f MiB)",
            " + ".join(spec.path.name for spec in desired),
            len(self._entries),
            self.unique_tensor_bytes() / (1024 * 1024),
        )

        try:
            from nunchaku.lora.common import compose_lora
        except ImportError as error:
            raise RuntimeError(
                "The installed Nunchaku backend does not provide the public "
                "nunchaku.lora.common.compose_lora API required for composed "
                "FLUX.2 Klein LoRAs."
            ) from error

        # Always start from the canonical flat graph tuple. Reusing prepared
        # subgroups would make floating-point 1-D sums grouping-dependent.
        prepared = compose_lora(
            [(spec.state_dict, spec.strength) for spec in desired]
        )
        if not isinstance(prepared, dict) or not prepared:
            raise RuntimeError("Nunchaku compose_lora() returned no prepared weights.")
        if not all(torch.is_tensor(tensor) for tensor in prepared.values()):
            raise RuntimeError("Nunchaku compose_lora() returned non-tensor weights.")

        self._entries[identity] = prepared
        logger.info(
            "Prepared LoRA cache: STORE %s (%d entries, %.2f MiB)",
            " + ".join(spec.path.name for spec in desired),
            len(self._entries),
            self.unique_tensor_bytes() / (1024 * 1024),
        )

        protected = {identity, self._active_identity}

        if self._evict(protected):
            logger.info(
                "Prepared LoRA cache: AFTER EVICT (%d entries, %.2f MiB)",
                len(self._entries),
                self.unique_tensor_bytes() / (1024 * 1024),
            )

        return prepared

    def set_active(self, identity: tuple[tuple, ...] | None) -> None:
        self._active_identity = identity if identity in self._entries else None
        self._evict({self._active_identity})

    def _evict(self, protected: set[tuple[tuple, ...] | None]) -> bool:
        evicted_any = False
        while len(self._entries) > self.max_entries:
            evicted = False
            for identity in tuple(self._entries):
                if identity not in protected:
                    logger.info(
                        "Prepared LoRA cache: EVICT %s",
                        " + ".join(Path(spec[1]).name for spec in identity),
                    )
                    del self._entries[identity]
                    evicted = True
                    evicted_any = True
                    break

            if not evicted:
                raise RuntimeError("Prepared LoRA cache cannot evict a protected entry.")

        return evicted_any

    def unique_tensor_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for prepared in self._entries.values():
            for tensor in prepared.values():
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    total += tensor.nbytes
        return total

    @property
    def identities(self) -> tuple[tuple[tuple, ...], ...]:
        return tuple(self._entries)


def _get_nunchaku_tensor_dictionaries(
    transformer: nn.Module,
) -> tuple[dict, ...]:
    states = []
    for name in NUNCHAKU_TENSOR_DICTIONARIES:
        state = getattr(transformer, name, None)
        if state is None:
            continue
        if not isinstance(state, dict):
            raise RuntimeError(
                f"Unsupported Nunchaku state: {name} must be a dict, got "
                f"{type(state).__name__}."
            )
        states.append(state)

    ranks = getattr(transformer, "_quantized_part_ranks", None)
    if ranks is not None and not isinstance(ranks, dict):
        raise RuntimeError(
            "Unsupported Nunchaku state: _quantized_part_ranks must be a dict."
        )
    if getattr(transformer, "offload", False):
        raise RuntimeError(
            "Nunchaku internal block offload must be disabled for this adapter."
        )
    return tuple(states)


def _iter_unique_live_tensors(
    base_model: nn.Module,
    transformer: nn.Module,
) -> Iterator[torch.Tensor]:
    seen: set[int] = set()
    for tensor in (*base_model.parameters(), *base_model.buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            yield tensor

    for state in _get_nunchaku_tensor_dictionaries(transformer):
        for tensor in state.values():
            if torch.is_tensor(tensor) and id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor


def calculate_live_model_size(
    base_model: nn.Module,
    transformer: nn.Module,
) -> int:
    return sum(tensor.nbytes for tensor in _iter_unique_live_tensors(base_model, transformer))


class SharedKleinLoraState:
    def __init__(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.lock = threading.RLock()
        self.active_identity: tuple[tuple, ...] = ()
        self.generation = 0
        self.invalid_reason: str | None = None
        self.current_size: int | None = None
        self._patchers = weakref.WeakSet()
        self.prepared_cache = KleinLoraPreparationCache()

    def require_lora_capabilities(self) -> None:
        for method in (
            "update_lora_params",
            "set_lora_strength",
            "reset_lora",
        ):
            if not callable(getattr(self.transformer, method, None)):
                raise RuntimeError(
                    "The installed Nunchaku FLUX.2 backend does not support "
                    f"branch-safe LoRA switching: missing {method}()."
                )

    def register_patcher(self, patcher) -> None:
        with self.lock:
            self._patchers.add(patcher)
            if self.current_size is not None:
                patcher.size = self.current_size

    def _refresh_accounting(self, old_size: int) -> None:
        patchers = list(self._patchers)
        if not patchers:
            raise RuntimeError("No live ModelPatcher is registered for this model.")

        base_models = {id(patcher.model): patcher.model for patcher in patchers}
        if len(base_models) != 1:
            raise RuntimeError(
                "Unsupported Nunchaku clone layout: LoRA branches must share "
                "one ComfyUI BaseModel."
            )
        base_model = next(iter(base_models.values()))
        loaded_memory = base_model.model_loaded_weight_memory
        if loaded_memory == old_size:
            fully_loaded = True
        elif loaded_memory == 0:
            fully_loaded = False
        else:
            raise RuntimeError(
                "Cannot change a Klein LoRA while ComfyUI reports partial "
                f"residency ({loaded_memory} of {old_size} bytes)."
            )

        new_size = calculate_live_model_size(base_model, self.transformer)
        for patcher in patchers:
            patcher.size = new_size
        base_model.model_loaded_weight_memory = new_size if fully_loaded else 0
        self.current_size = new_size

    @staticmethod
    def _log_desired_loras(
        desired: tuple[KleinLoraSpec, ...],
    ) -> None:
        if not desired:
            logger.info("Desired LoRAs:\n  (none)")
            return
        entries = "\n".join(
            f"  [{index}] {spec.path.name} @ {spec.strength:.3f}"
            for index, spec in enumerate(desired)
        )
        logger.info("Desired LoRAs:\n%s", entries)

    def ensure(self, desired: tuple[KleinLoraSpec, ...]) -> None:
        with self.lock:
            self._ensure_locked(desired)

    def _ensure_locked(self, desired: tuple[KleinLoraSpec, ...]) -> None:
        if self.invalid_reason is not None:
            raise RuntimeError(
                "The shared Nunchaku transformer LoRA state is invalid; "
                "reload the model before sampling. " + self.invalid_reason
            )

        identity = tuple(spec.identity for spec in desired)
        if identity == self.active_identity:
            return
        self._log_desired_loras(desired)
        self.require_lora_capabilities()
        if self.current_size is None:
            raise RuntimeError("Nunchaku LoRA accounting was not initialized.")

        old_size = self.current_size
        prepared = None
        use_composition = len(desired) > 1 or (
            len(desired) == 1 and desired[0].strength <= 0
        )
        if use_composition:
            # Preparation happens before backend mutation. A compose failure
            # therefore leaves physical weights and accounting untouched.
            prepared = self.prepared_cache.prepare(desired)

        try:
            if not desired:
                logger.info("Transition: RESET")
                self.transformer.reset_lora()
            elif (
                not use_composition
                and len(self.active_identity) == 1
                and self.active_identity[0][:-1] == desired[0].source_identity
            ):
                logger.info("Transition: SET_STRENGTH")
                self.transformer.set_lora_strength(desired[0].strength)
            elif use_composition:
                logger.info("Transition: COMPOSE %d", len(desired))
                self.transformer.update_lora_params(prepared, strength=1.0)
            else:
                logger.info("Transition: APPLY")
                self.transformer.update_lora_params(
                    desired[0].state_dict,
                    strength=desired[0].strength,
                )
            self._refresh_accounting(old_size)
        except Exception as transition_error:
            try:
                self.transformer.reset_lora()
                self._refresh_accounting(old_size)
            except Exception as rollback_error:
                self.invalid_reason = (
                    f"LoRA transition failed ({transition_error!r}) and "
                    f"rollback failed ({rollback_error!r})."
                )
                raise RuntimeError(self.invalid_reason) from transition_error
            self.active_identity = ()
            self.prepared_cache.set_active(None)
            raise RuntimeError(
                "Nunchaku LoRA transition failed; original weights were "
                "restored and the requested forward was cancelled."
            ) from transition_error

        self.active_identity = identity
        self.prepared_cache.set_active(identity if use_composition else None)
        self.generation += 1


def get_or_create_shared_lora_state(
    transformer: nn.Module,
) -> SharedKleinLoraState:
    state = getattr(transformer, _SHARED_LORA_STATE_ATTRIBUTE, None)
    if state is None:
        state = SharedKleinLoraState(transformer)
        setattr(transformer, _SHARED_LORA_STATE_ATTRIBUTE, state)
    elif not isinstance(state, SharedKleinLoraState):
        raise RuntimeError(
            f"Transformer attribute {_SHARED_LORA_STATE_ATTRIBUTE!r} is "
            "already owned by incompatible code."
        )
    return state


class NunchakuFlux2KleinAdapter(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        *,
        in_channels: int,
        context_dim: int,
        patch_size: int,
        axes_dim: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        _get_nunchaku_tensor_dictionaries(transformer)
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if len(axes_dim) != 4:
            raise ValueError(
                f"FLUX.2 Klein requires four RoPE axes, got {list(axes_dim)}."
            )

        self.transformer = transformer
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.patch_size = patch_size
        self.axes_dim = axes_dim
        self.dtype = dtype
        self.shared_lora_state = get_or_create_shared_lora_state(transformer)

    def _apply(self, fn, recurse: bool = True):
        # Nunchaku 1.2.1 keeps LoRA originals/caches in private plain dicts.
        # Track registered aliases so every unique tensor is transformed once.
        with self.shared_lora_state.lock:
            registered_before = [*self.parameters(), *self.buffers()]
            result = super()._apply(fn, recurse=recurse)
            registered_after = [*self.parameters(), *self.buffers()]
            moved = {
                id(before): after
                for before, after in zip(
                    registered_before,
                    registered_after,
                    strict=True,
                )
            }

            for state in _get_nunchaku_tensor_dictionaries(self.transformer):
                for key, tensor in list(state.items()):
                    if not torch.is_tensor(tensor):
                        continue
                    moved_tensor = moved.get(id(tensor))
                    if moved_tensor is None:
                        moved_tensor = fn(tensor)
                        moved[id(tensor)] = moved_tensor
                    state[key] = moved_tensor
            return result

    def _pack_latents(
        self,
        x: torch.Tensor,
        *,
        image_index: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        batch, channels, height, width = x.shape
        patch = self.patch_size
        padded_height = ((height + patch - 1) // patch) * patch
        padded_width = ((width + patch - 1) // patch) * patch
        if padded_height != height or padded_width != width:
            x = F.pad(x, (0, padded_width - width, 0, padded_height - height))

        grid_height = padded_height // patch
        grid_width = padded_width // patch
        image = (
            x.reshape(
                batch,
                channels,
                grid_height,
                patch,
                grid_width,
                patch,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid_height * grid_width, channels * patch * patch)
        )

        image_ids = torch.zeros(
            (grid_height, grid_width, len(self.axes_dim)),
            device=x.device,
            dtype=torch.float32,
        )
        image_ids[..., 0] = image_index
        image_ids[..., 1] = torch.arange(
            grid_height,
            device=x.device,
            dtype=torch.float32,
        ).view(grid_height, 1)
        image_ids[..., 2] = torch.arange(
            grid_width,
            device=x.device,
            dtype=torch.float32,
        ).view(1, grid_width)
        return (
            image,
            image_ids.reshape(grid_height * grid_width, len(self.axes_dim)),
            grid_height,
            grid_width,
        )

    def _unpack_latents(
        self,
        tokens: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        patch = self.patch_size
        output = (
            tokens.reshape(
                batch,
                grid_height,
                grid_width,
                self.in_channels,
                patch,
                patch,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(
                batch,
                self.in_channels,
                grid_height * patch,
                grid_width * patch,
            )
        )
        return output[:, :, :height, :width]

    def _validate_transformer_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> None:
        batch, image_tokens, _ = hidden_states.shape
        text_tokens = encoder_hidden_states.shape[1]
        errors = []
        if encoder_hidden_states.shape[0] != batch:
            errors.append(
                "hidden_states and encoder_hidden_states batches differ "
                f"({batch} versus {encoder_hidden_states.shape[0]})"
            )
        if timestep.ndim != 1 or timestep.shape[0] != batch:
            errors.append(
                f"timestep must have shape [{batch}], got {list(timestep.shape)}"
            )
        if img_ids.shape != (image_tokens, len(self.axes_dim)):
            errors.append(
                "img_ids must be an unbatched position table shaped "
                f"[{image_tokens}, {len(self.axes_dim)}], got {list(img_ids.shape)}"
            )
        if txt_ids.shape != (text_tokens, len(self.axes_dim)):
            errors.append(
                "txt_ids must be an unbatched position table shaped "
                f"[{text_tokens}, {len(self.axes_dim)}], got {list(txt_ids.shape)}"
            )
        tensors = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
        wrong_devices = {
            name: str(tensor.device)
            for name, tensor in tensors.items()
            if tensor.device != hidden_states.device
        }
        if wrong_devices:
            errors.append(
                f"all transformer inputs must be on {hidden_states.device}; "
                f"mismatches: {wrong_devices}"
            )
        if errors:
            raise ValueError(
                "Unsafe Nunchaku FLUX.2 input layout; refusing to enter fused "
                "CUDA kernels: " + "; ".join(errors) + "."
            )

    def _pack_reference_latents(
        self,
        x: torch.Tensor,
        image: torch.Tensor,
        image_ids: torch.Tensor,
        ref_latents: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ref_latents is None:
            return image, image_ids
        if not isinstance(ref_latents, list) or not ref_latents:
            raise TypeError(
                "FLUX.2 Klein ref_latents must be a non-empty list of "
                "BCHW latent tensors."
            )

        batch = x.shape[0]
        packed_references = []
        reference_ids = []
        for ref_index, reference in enumerate(ref_latents, start=1):
            if not torch.is_tensor(reference):
                raise TypeError(
                    f"Reference {ref_index} must be a torch.Tensor, got "
                    f"{type(reference).__name__}."
                )
            if reference.ndim != 4 or reference.shape[1] != self.in_channels:
                raise ValueError(
                    f"Reference {ref_index} must be BCHW with "
                    f"{self.in_channels} channels, got {list(reference.shape)}."
                )
            if reference.shape[0] != batch:
                raise ValueError(
                    f"Reference {ref_index} batch must match the effective "
                    f"sampling batch {batch}, got {reference.shape[0]}."
                )
            if reference.shape[-2] < 1 or reference.shape[-1] < 1:
                raise ValueError(
                    f"Reference {ref_index} spatial dimensions must be "
                    f"positive, got {list(reference.shape[-2:])}."
                )
            if reference.device != x.device or reference.dtype != x.dtype:
                raise ValueError(
                    f"Reference {ref_index} must match generated latent "
                    f"device/dtype {x.device}/{x.dtype}, got "
                    f"{reference.device}/{reference.dtype}."
                )
            packed, ids, _, _ = self._pack_latents(
                reference,
                image_index=REFERENCE_IMAGE_INDEX_STRIDE * ref_index,
            )
            packed_references.append(packed)
            reference_ids.append(ids)

        return (
            torch.cat([image, *packed_references], dim=1),
            torch.cat([image_ids, *reference_ids], dim=0),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y=None,
        guidance=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                "Expected a BCHW FLUX.2 latent with "
                f"{self.in_channels} channels, got shape {list(x.shape)}."
            )
        if context is None or context.ndim != 3 or context.shape[-1] != self.context_dim:
            shape = None if context is None else list(context.shape)
            raise ValueError(
                "Expected text context shaped [batch, tokens, "
                f"{self.context_dim}], got {shape}."
            )
        if timestep is None:
            raise ValueError("FLUX.2 Klein requires a timestep tensor.")
        if control is not None:
            raise NotImplementedError("ControlNet is not supported by this loader.")

        batch, _, height, width = x.shape
        image, image_ids, grid_height, grid_width = self._pack_latents(x)
        generated_tokens = image.shape[1]
        ref_latents = kwargs.get("ref_latents")
        if ref_latents is not None:
            ref_method = kwargs.get("ref_latents_method")
            # Current ComfyUI Flux2 uses "index". None means its model-config
            # default, which is also "index" for the validated Klein 9B path.
            if ref_method not in (None, "index"):
                raise NotImplementedError(
                    "This adapter supports only ComfyUI's FLUX.2 Klein "
                    f"reference method 'index', got {ref_method!r}."
                )

        text_ids = torch.zeros(
            (context.shape[1], len(self.axes_dim)),
            device=x.device,
            dtype=torch.float32,
        )
        text_ids[..., 3] = torch.arange(
            context.shape[1],
            device=x.device,
            dtype=torch.float32,
        )

        self._validate_transformer_inputs(
            image,
            context,
            timestep,
            image_ids,
            text_ids,
        )

        # Nunchaku 1.2.1 discards the batch axis of 3-D position IDs, while its
        # fused rotary kernel flattens B*S. Dispatching one sample at a time is
        # required for correct CFG batches until the backend supports batched
        # Flux.2 rotary embeddings end to end.
        desired = ()
        if transformer_options is not None:
            desired = transformer_options.get(LORA_SPEC_OPTION, ())
        if not isinstance(desired, tuple) or not all(
            isinstance(spec, KleinLoraSpec) for spec in desired
        ):
            raise TypeError(
                f"{LORA_SPEC_OPTION} must contain a tuple of KleinLoraSpec, got "
                f"{type(desired).__name__}."
            )

        ref_weight_spec = None
        if transformer_options is not None:
            ref_weight_spec = transformer_options.get(REF_LATENT_WEIGHT_OPTION)
        if ref_weight_spec is not None and not isinstance(
            ref_weight_spec, KleinRefLatentWeightSpec
        ):
            raise TypeError(
                f"{REF_LATENT_WEIGHT_OPTION} must contain "
                "KleinRefLatentWeightSpec."
            )
        if ref_weight_spec is not None:
            if not isinstance(ref_latents, list) or not ref_latents:
                raise ValueError(
                    "Nunchaku FLUX.2 Klein Ref Latent Weight requires a "
                    "non-empty runtime reference list."
                )
            if not 0 <= ref_weight_spec.reference_index < len(ref_latents):
                raise IndexError(
                    "Ref Latent Weight reference_index "
                    f"{ref_weight_spec.reference_index} is out of range for "
                    f"{len(ref_latents)} runtime references."
                )

        # The transformer weights are shared by ModelPatcher clones. Keep the
        # lock through every serialized CFG/batch call so no sibling branch can
        # switch physical LoRA state during this logical adapter forward.
        with self.shared_lora_state.lock:
            self.shared_lora_state.ensure(desired)
            logger.debug(
                "Ref Latent Weight: desired_loras=%d active_identity=%s",
                len(desired),
                self.shared_lora_state.active_identity,
            )
            def predict(references: list[torch.Tensor] | None) -> torch.Tensor:
                logger.debug(
                    "predict(): batch=%d refs=%d generated_tokens=%d",
                    batch,
                    0 if references is None else len(references),
                    generated_tokens,
                )
                hidden_states, img_ids = self._pack_reference_latents(
                    x, image, image_ids, references
                )
                self._validate_transformer_inputs(
                    hidden_states, context, timestep, img_ids, text_ids
                )
                outputs = []
                for index in range(batch):
                    sample = self.transformer(
                        hidden_states=hidden_states[index : index + 1],
                        encoder_hidden_states=context[index : index + 1],
                        timestep=timestep[index : index + 1],
                        img_ids=img_ids,
                        txt_ids=text_ids,
                        guidance=None,
                    ).sample
                    if (
                        sample.ndim != 3
                        or sample.shape[0] != 1
                        or sample.shape[1] != hidden_states.shape[1]
                        or sample.shape[2]
                        != self.in_channels * self.patch_size**2
                    ):
                        raise ValueError(
                            "Nunchaku FLUX.2 returned an unsafe reference-edit "
                            "token layout: expected "
                            f"[1, {hidden_states.shape[1]}, "
                            f"{self.in_channels * self.patch_size**2}], got "
                            f"{list(sample.shape)}."
                        )
                    # The backend returns generated tokens followed by reference
                    # tokens. Discard reference outputs before reconstruction.
                    outputs.append(sample[:, :generated_tokens])
                return torch.cat(outputs, dim=0)


            if ref_weight_spec is None:
                logger.debug(
                    "Ref Latent Weight: no spec; path=ordinary refs=%d",
                    0 if ref_latents is None else len(ref_latents),
                )
                output = predict(ref_latents)

            elif ref_weight_spec.weight == 1.0:
                logger.info(
                    "Ref Latent Weight: index=%d weight=1.000 path=full-reference one-pass",
                    ref_weight_spec.reference_index,
                )
                output = predict(ref_latents)

            else:
                remaining = [
                    reference
                    for index, reference in enumerate(ref_latents)
                    if index != ref_weight_spec.reference_index
                ]

                logger.debug(
                    "Ref Latent Weight: index=%d weight=%.3f pass=WITHOUT selected "
                    "refs=%d->%d",
                    ref_weight_spec.reference_index,
                    ref_weight_spec.weight,
                    len(ref_latents),
                    len(remaining),
                )
                without_selected = predict(remaining or None)

                if ref_weight_spec.weight == 0.0:
                    logger.info(
                        "Ref Latent Weight: index=%d weight=0.000 path=ablation one-pass",
                        ref_weight_spec.reference_index,
                    )
                    output = without_selected
                else:
                    logger.debug(
                        "Ref Latent Weight: index=%d weight=%.3f pass=WITH selected refs=%d",
                        ref_weight_spec.reference_index,
                        ref_weight_spec.weight,
                        len(ref_latents),
                    )
                    with_selected = predict(ref_latents)

                    logger.debug(
                        "Ref Latent Weight: index=%d weight=%.3f combine=prediction-residual",
                        ref_weight_spec.reference_index,
                        ref_weight_spec.weight,
                    )

                    output = (
                        without_selected.float()
                        + ref_weight_spec.weight
                        * (with_selected.float() - without_selected.float())
                    ).to(dtype=with_selected.dtype)
            return self._unpack_latents(
                output,
                grid_height=grid_height,
                grid_width=grid_width,
                height=height,
                width=width,
            )
