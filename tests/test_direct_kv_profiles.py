import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_direct_profiles_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_direct_profiles_test.nodes.enhancer.ref_latent_controller_direct import (
    KleinRefLatentControllerKVCallback,
)
from nunchaku_klein_direct_profiles_test.nodes.enhancer.ref_latent_weight_direct import (
    KleinRefLatentWeightKVCallback,
)
from nunchaku_klein_direct_profiles_test.nodes.enhancer.text_ref_balance_direct import (
    KleinTextRefBalanceKVCallback,
)


PROFILES = {
    "4B": {"heads": 24, "head_dim": 128, "double": 5, "single": 20},
    "9B": {"heads": 32, "head_dim": 128, "double": 8, "single": 24},
}


def metadata(profile, block_type, block_index, refs=(6, 4)):
    text, generated = 5, 6
    logical_image = generated + sum(refs)
    if block_type == "double":
        padded_text, padded_image = 8, 16
    else:
        padded_text, padded_image = text, 19
    values = PROFILES[profile]
    return SimpleNamespace(
        block_type=block_type,
        block_index=block_index,
        text_token_count=text,
        generated_token_count=generated,
        reference_token_counts=refs,
        reference_spatial_shapes=tuple((2, count // 2) for count in refs),
        logical_image_token_count=logical_image,
        padded_text_token_count=padded_text,
        padded_image_token_count=padded_image,
        packed_sequence_length=padded_text + padded_image,
        batch_size=1,
        head_count=values["heads"],
        head_dimension=values["head_dim"],
    )


def tensors(info, dtype=torch.float32):
    shape = (
        info.batch_size,
        info.head_count,
        info.packed_sequence_length,
        info.head_dimension,
    )
    query = torch.ones(shape, dtype=dtype)
    return query, query.clone().mul_(2), query.clone().mul_(3)


def ranges(info):
    image_start = (
        info.padded_text_token_count
        if info.block_type == "double"
        else info.text_token_count
    )
    generated = (image_start, image_start + info.generated_token_count)
    references = []
    position = generated[1]
    for count in info.reference_token_counts:
        references.append((position, position + count))
        position += count
    return (0, info.text_token_count), generated, tuple(references)


class DirectKVProfileTests(unittest.TestCase):
    def test_real_callbacks_accept_every_4b_and_9b_block_shape(self):
        for profile, values in PROFILES.items():
            for block_type in ("double", "single"):
                for block_index in range(values[block_type]):
                    with self.subTest(profile=profile, block_type=block_type, block=block_index):
                        info = metadata(profile, block_type, block_index)
                        q, k, v = tensors(info, torch.float16)
                        before = tuple(item.clone() for item in (q, k, v))
                        KleinRefLatentWeightKVCallback(0, 1.0)(q, k, v, info)
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip((q, k, v), before, strict=True)))
                        self.assertEqual(q.shape[1:], (values["heads"], info.packed_sequence_length, 128))

    def test_ordered_composition_changes_only_intended_text_and_references(self):
        for profile in PROFILES:
            for block_type in ("double", "single"):
                with self.subTest(profile=profile, block_type=block_type):
                    info = metadata(profile, block_type, 0)
                    q, k, v = tensors(info)
                    qb, kb, vb = (item.clone() for item in (q, k, v))
                    text, generated, references = ranges(info)
                    callbacks = (
                        KleinTextRefBalanceKVCallback(0.5, 0.8),
                        KleinRefLatentWeightKVCallback(0, 0.5),
                        KleinRefLatentControllerKVCallback(1.25, 1, "none", 0.5),
                    )
                    for callback in callbacks:
                        callback(q, k, v, info)
                    self.assertTrue(torch.equal(q, qb))
                    self.assertTrue(torch.equal(k[:, :, text[0]:text[1]], kb[:, :, text[0]:text[1]] * 0.5))
                    self.assertTrue(torch.equal(k[:, :, generated[0]:generated[1]], kb[:, :, generated[0]:generated[1]]))
                    self.assertTrue(torch.equal(k[:, :, references[0][0]:references[0][1]], kb[:, :, references[0][0]:references[0][1]] * 0.4))
                    self.assertTrue(torch.equal(k[:, :, references[1][0]:references[1][1]], kb[:, :, references[1][0]:references[1][1]]))
                    self.assertTrue(torch.equal(v / vb, k / kb))

    def test_no_reference_contract_is_consumer_specific(self):
        info = metadata("4B", "single", 0, refs=())
        q, k, v = tensors(info)
        before = tuple(item.clone() for item in (q, k, v))
        KleinRefLatentControllerKVCallback(2.0, 0, "none", 0.5)(q, k, v, info)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((q, k, v), before, strict=True)))
        KleinTextRefBalanceKVCallback(0.5, 0.25)(q, k, v, info)
        self.assertTrue(torch.equal(k[:, :, :info.text_token_count], before[1][:, :, :info.text_token_count] * 0.5))
        with self.assertRaises(IndexError):
            KleinRefLatentWeightKVCallback(0, 0.5)(q, k, v, info)

    def test_spatial_controller_uses_runtime_non_square_reference_shape(self):
        info = metadata("4B", "double", 4, refs=(6,))
        q, k, v = tensors(info)
        kb = k.clone()
        _, _, references = ranges(info)
        start, end = references[0]
        KleinRefLatentControllerKVCallback(1.0, 0, "left_right", 1.0)(q, k, v, info)
        expected = torch.tensor([1.0, 0.5, 0.0, 1.0, 0.5, 0.0]).view(1, 1, -1, 1)
        self.assertTrue(torch.equal(k[:, :, start:end], kb[:, :, start:end] * expected))

    def test_malformed_head_batch_and_block_metadata_fail_before_mutation(self):
        valid = metadata("4B", "double", 0)
        for field, value in (("head_count", 32), ("head_dimension", 64), ("batch_size", 2), ("block_index", -1)):
            with self.subTest(field=field):
                info = SimpleNamespace(**vars(valid))
                setattr(info, field, value)
                q, k, v = tensors(valid)
                before = tuple(item.clone() for item in (q, k, v))
                with self.assertRaises(ValueError):
                    KleinRefLatentWeightKVCallback(0, 0.5)(q, k, v, info)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip((q, k, v), before, strict=True)))


if __name__ == "__main__":
    unittest.main()
