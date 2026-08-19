"""Shared strict token-layout validation for direct FLUX.2 K/V callbacks."""

import torch


def resolve_attention_token_ranges(metadata, packed_sequence_length, *, name):
    integer_fields = (
        "block_index",
        "text_token_count",
        "generated_token_count",
        "logical_image_token_count",
        "padded_text_token_count",
        "padded_image_token_count",
        "packed_sequence_length",
        "batch_size",
        "head_count",
        "head_dimension",
    )
    missing = [
        field
        for field in ("block_type", "reference_token_counts", *integer_fields)
        if not hasattr(metadata, field)
    ]
    if missing:
        raise TypeError(f"{name} callback metadata is missing: " + ", ".join(missing))
    for field in integer_fields:
        value = getattr(metadata, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} metadata {field} must be a non-negative integer.")

    reference_counts = metadata.reference_token_counts
    if not isinstance(reference_counts, tuple) or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in reference_counts
    ):
        raise ValueError(
            f"{name} reference_token_counts must be an ordered tuple of "
            "positive integers."
        )
    expected_image = metadata.generated_token_count + sum(reference_counts)
    if metadata.logical_image_token_count != expected_image:
        raise ValueError(
            f"{name} logical image count is inconsistent: "
            f"{metadata.logical_image_token_count} != {expected_image}."
        )
    if packed_sequence_length != metadata.packed_sequence_length:
        raise ValueError(
            f"{name} packed sequence mismatch: tensor has {packed_sequence_length} "
            f"tokens but metadata reports {metadata.packed_sequence_length}."
        )
    if (
        metadata.packed_sequence_length
        != metadata.padded_text_token_count + metadata.padded_image_token_count
    ):
        raise ValueError(
            f"{name} padded text/image counts do not match the packed sequence length."
        )

    if metadata.block_type == "double":
        if metadata.padded_text_token_count < metadata.text_token_count:
            raise ValueError(f"{name} double-stream text padding is invalid.")
        if metadata.padded_image_token_count < expected_image:
            raise ValueError(f"{name} double-stream image padding is invalid.")
        image_start = metadata.padded_text_token_count
    elif metadata.block_type == "single":
        if metadata.padded_text_token_count != metadata.text_token_count:
            raise ValueError(
                f"{name} single-stream text count must remain logical; "
                "single-stream padding is trailing."
            )
        if metadata.text_token_count + expected_image > metadata.packed_sequence_length:
            raise ValueError(f"{name} single-stream logical sequence is invalid.")
        image_start = metadata.text_token_count
    else:
        raise ValueError(f"{name} received unsupported block_type {metadata.block_type!r}.")

    generated_range = (
        image_start,
        image_start + metadata.generated_token_count,
    )
    reference_ranges = []
    position = generated_range[1]
    for count in reference_counts:
        reference_ranges.append((position, position + count))
        position += count
    logical_image_end = image_start + expected_image
    if position != logical_image_end or logical_image_end > packed_sequence_length:
        raise ValueError(
            f"{name} logical image range [{image_start}, {logical_image_end}) "
            f"is outside packed length {packed_sequence_length}."
        )
    return (0, metadata.text_token_count), generated_range, tuple(reference_ranges)


def validate_attention_kv_layout(query, key, value, metadata, *, name):
    tensors = {"query": query, "key": key, "value": value}
    for tensor_name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 4:
            shape = None if not torch.is_tensor(tensor) else list(tensor.shape)
            raise ValueError(
                f"{name} {tensor_name} must be a rank-4 tensor, got {shape}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} {tensor_name} must be contiguous.")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            f"{name} requires identical Q/K/V shapes, got "
            f"Q={list(query.shape)}, K={list(key.shape)}, V={list(value.shape)}."
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError(
            f"{name} requires identical Q/K/V dtypes, got "
            f"Q={query.dtype}, K={key.dtype}, V={value.dtype}."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"{name} requires identical Q/K/V devices, got "
            f"Q={query.device}, K={key.device}, V={value.device}."
        )
    ranges = resolve_attention_token_ranges(metadata, query.shape[2], name=name)
    expected = (
        metadata.batch_size,
        metadata.head_count,
        metadata.packed_sequence_length,
        metadata.head_dimension,
    )
    if query.shape != expected:
        raise ValueError(
            f"{name} Q/K/V shape {list(query.shape)} does not match callback "
            f"metadata {list(expected)}."
        )
    return ranges


__all__ = ["resolve_attention_token_ranges", "validate_attention_kv_layout"]
