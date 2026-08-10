import re

import torch


_PAIR_SUFFIXES = ("lora_A.weight", "lora_B.weight")
_SPLIT_SUFFIXES = ("lora.down.weight", "lora.up.weight")

_CANONICAL_TARGET = re.compile(
    r"^(?:diffusion_model\.)?"
    r"(?:(?:double_blocks\.\d+\."
    r"(?:img_attn\.(?:qkv|proj)|txt_attn\.(?:qkv|proj)|"
    r"img_mlp\.(?:0|2)|txt_mlp\.(?:0|2)))|"
    r"(?:single_blocks\.\d+\.linear(?:1|2)))$"
)
_SPLIT_TARGET = re.compile(
    r"^transformer\."
    r"(?P<blocks>transformer_blocks|single_transformer_blocks)\."
    r"(?P<index>\d+)\.(?P<target>.+)$"
)

_DOUBLE_DIRECT_TARGETS = {
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}
_SINGLE_DIRECT_TARGETS = {
    "attn.to_qkv_mlp_proj": "linear1",
    "attn.to_out": "linear2",
}
_QKV_TARGETS = {
    "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn.qkv": (
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
    ),
}


def _validate_pair(
    state_dict: dict[str, torch.Tensor],
    target: str,
    suffixes: tuple[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    first_key = f"{target}.{suffixes[0]}"
    second_key = f"{target}.{suffixes[1]}"
    missing = [key for key in (first_key, second_key) if key not in state_dict]
    if missing:
        raise ValueError(
            f"Incomplete Klein LoRA pair for {target!r}; missing keys: {missing}."
        )

    first = state_dict[first_key]
    second = state_dict[second_key]
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(
            f"Klein LoRA pair {target!r} must contain two matrices, got "
            f"{list(first.shape)} and {list(second.shape)}."
        )
    if any(dimension <= 0 for dimension in (*first.shape, *second.shape)):
        raise ValueError(
            f"Klein LoRA pair {target!r} has an empty dimension: "
            f"{list(first.shape)} and {list(second.shape)}."
        )
    if first.shape[0] != second.shape[1]:
        raise ValueError(
            f"Klein LoRA pair {target!r} has inconsistent rank: "
            f"{list(first.shape)} and {list(second.shape)}."
        )
    if first.dtype != second.dtype:
        raise ValueError(
            f"Klein LoRA pair {target!r} has mismatched dtypes: "
            f"{first.dtype} and {second.dtype}."
        )
    if first.device.type != "cpu" or second.device.type != "cpu":
        raise ValueError(f"Klein LoRA pair {target!r} must be loaded on CPU.")
    return first, second


def _canonical_targets(state_dict: dict[str, torch.Tensor]) -> set[str] | None:
    targets = set()
    semantic_targets = set()
    for key in state_dict:
        matched_suffix = next(
            (suffix for suffix in _PAIR_SUFFIXES if key.endswith(f".{suffix}")),
            None,
        )
        if matched_suffix is None:
            return None
        target = key[: -(len(matched_suffix) + 1)]
        if _CANONICAL_TARGET.fullmatch(target) is None:
            return None
        semantic_target = target.removeprefix("diffusion_model.")
        if semantic_target in semantic_targets and target not in targets:
            raise ValueError(
                f"Ambiguous duplicate canonical Klein LoRA target: "
                f"{semantic_target!r}."
            )
        targets.add(target)
        semantic_targets.add(semantic_target)
    return targets


def _split_targets(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[str, int, str]]:
    targets = {}
    offending = []
    for key in state_dict:
        matched_suffix = next(
            (
                suffix
                for suffix in (*_SPLIT_SUFFIXES, *_PAIR_SUFFIXES)
                if key.endswith(f".{suffix}")
            ),
            None,
        )
        if matched_suffix is None:
            offending.append(key)
            continue
        target = key[: -(len(matched_suffix) + 1)]
        match = _SPLIT_TARGET.fullmatch(target)
        if match is None:
            offending.append(key)
            continue
        targets[target] = (
            match.group("blocks"),
            int(match.group("index")),
            match.group("target"),
        )
    if offending:
        raise ValueError(
            "Unsupported or mixed Klein LoRA keys: " + repr(sorted(offending))
        )
    return targets


def _source_pair(
    state_dict: dict[str, torch.Tensor],
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, str]]:
    complete = [
        suffixes
        for suffixes in (_SPLIT_SUFFIXES, _PAIR_SUFFIXES)
        if all(f"{target}.{suffix}" in state_dict for suffix in suffixes)
    ]
    present = [
        f"{target}.{suffix}"
        for suffix in (*_SPLIT_SUFFIXES, *_PAIR_SUFFIXES)
        if f"{target}.{suffix}" in state_dict
    ]
    if not complete and present:
        present_suffixes = {
            key[len(target) + 1 :] for key in present
        }
        selected = next(
            (
                suffixes
                for suffixes in (_SPLIT_SUFFIXES, _PAIR_SUFFIXES)
                if present_suffixes.intersection(suffixes)
            ),
            None,
        )
        if selected is not None:
            missing = [
                f"{target}.{suffix}"
                for suffix in selected
                if f"{target}.{suffix}" not in state_dict
            ]
            raise ValueError(
                f"Incomplete Klein LoRA pair for {target!r}; missing keys: "
                f"{missing}."
            )
    if len(complete) != 1 or len(present) != 2:
        raise ValueError(
            f"Klein LoRA source target {target!r} must contain exactly one "
            f"complete down/up or A/B pair; found keys: {present}."
        )
    suffixes = complete[0]
    first, second = _validate_pair(state_dict, target, suffixes)
    return first, second, suffixes


def _fuse_qkv(
    state_dict: dict[str, torch.Tensor],
    source_targets: tuple[str, str, str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[str, str], tuple[str, str], tuple[str, str]],
]:
    pairs_with_suffixes = [_source_pair(state_dict, target) for target in source_targets]
    pairs = [(first, second) for first, second, _suffixes in pairs_with_suffixes]
    inputs = {pair[0].shape[1] for pair in pairs}
    dtypes = {tensor.dtype for pair in pairs for tensor in pair}
    if len(inputs) != 1:
        raise ValueError(
            f"Split Q/K/V LoRA inputs differ for {source_targets!r}: "
            f"{[pair[0].shape[1] for pair in pairs]}."
        )
    if len(dtypes) != 1:
        raise ValueError(
            f"Split Q/K/V LoRA dtypes differ for {source_targets!r}: "
            f"{sorted(map(str, dtypes))}."
        )
    fused_a = torch.cat([pair[0] for pair in pairs], dim=0).contiguous()
    fused_b = torch.block_diag(*[pair[1] for pair in pairs]).contiguous()
    return fused_a, fused_b, tuple(
        suffixes for _first, _second, suffixes in pairs_with_suffixes
    )


def normalize_klein_lora_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a strictly validated BFL-style Klein LoRA state dictionary."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Klein LoRA state must be a non-empty dictionary.")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Klein LoRA state keys must be strings.")
    if not all(torch.is_tensor(tensor) for tensor in state_dict.values()):
        raise ValueError("Klein LoRA state contains non-tensor weights.")

    canonical_targets = _canonical_targets(state_dict)
    if canonical_targets is not None:
        for target in sorted(canonical_targets):
            _validate_pair(state_dict, target, _PAIR_SUFFIXES)
        return state_dict

    split_targets = _split_targets(state_dict)
    output: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for source_target, (blocks, index, target) in split_targets.items():
        grouped.setdefault((blocks, index), {})[target] = source_target

    for (blocks, index), targets in sorted(grouped.items()):
        if blocks == "transformer_blocks":
            for canonical_target, source_names in _QKV_TARGETS.items():
                present = [name for name in source_names if name in targets]
                if present and len(present) != len(source_names):
                    missing = [name for name in source_names if name not in targets]
                    raise ValueError(
                        f"Incomplete split Q/K/V group for double block {index} "
                        f"{canonical_target!r}; missing targets: {missing}."
                    )
                if present:
                    sources = tuple(targets[name] for name in source_names)
                    fused_a, fused_b, source_suffixes = _fuse_qkv(
                        state_dict, sources
                    )
                    destination = (
                        f"diffusion_model.double_blocks.{index}.{canonical_target}"
                    )
                    output[f"{destination}.lora_A.weight"] = fused_a
                    output[f"{destination}.lora_B.weight"] = fused_b
                    for source, suffixes in zip(sources, source_suffixes):
                        consumed.update(
                            f"{source}.{suffix}" for suffix in suffixes
                        )

            for source_name, canonical_target in _DOUBLE_DIRECT_TARGETS.items():
                source = targets.get(source_name)
                if source is None:
                    continue
                down, up, suffixes = _source_pair(state_dict, source)
                destination = (
                    f"diffusion_model.double_blocks.{index}.{canonical_target}"
                )
                output[f"{destination}.lora_A.weight"] = down
                output[f"{destination}.lora_B.weight"] = up
                consumed.update(f"{source}.{suffix}" for suffix in suffixes)
        else:
            for source_name, canonical_target in _SINGLE_DIRECT_TARGETS.items():
                source = targets.get(source_name)
                if source is None:
                    continue
                down, up, suffixes = _source_pair(state_dict, source)
                destination = (
                    f"diffusion_model.single_blocks.{index}.{canonical_target}"
                )
                output[f"{destination}.lora_A.weight"] = down
                output[f"{destination}.lora_B.weight"] = up
                consumed.update(f"{source}.{suffix}" for suffix in suffixes)

    unconsumed = sorted(set(state_dict) - consumed)
    if unconsumed:
        raise ValueError(
            "Unsupported or unconsumed split-QKV Klein LoRA keys: "
            + repr(unconsumed)
        )
    if not output:
        raise ValueError("Klein LoRA contains no supported canonical targets.")
    return output
