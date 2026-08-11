import importlib.util
import pathlib
import sys
import unittest

import torch
from torch import nn


TARGET = pathlib.Path(__file__).resolve().parents[1]
COMFY = pathlib.Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY))
spec = importlib.util.spec_from_file_location(
    "nunchaku_klein_attention_plumbing_test",
    TARGET / "__init__.py",
    submodule_search_locations=[str(TARGET)],
)
PACKAGE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PACKAGE
spec.loader.exec_module(PACKAGE)

from nunchaku_klein_attention_plumbing_test.models.klein_wrapper import (
    ATTENTION_CALLBACKS_OPTION,
    Flux2AttentionCallbacks,
    NunchakuFlux2KleinAdapter,
    REF_LATENT_CONTROLLER_DIRECT_OPTION,
)


class Result:
    def __init__(self, sample):
        self.sample = sample


class RecordingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.offload = False
        self.kwargs = []

    def forward(self, *, hidden_states, joint_attention_kwargs=None, **kwargs):
        self.kwargs.append(joint_attention_kwargs)
        return Result(hidden_states.clone())


def adapter_and_transformer():
    transformer = RecordingTransformer()
    adapter = NunchakuFlux2KleinAdapter(
        transformer,
        in_channels=2,
        context_dim=4,
        patch_size=1,
        axes_dim=(1, 1, 1, 1),
        dtype=torch.float32,
    )
    return adapter, transformer


def run(adapter, *, batch=1, references=(), callbacks=None):
    options = {}
    if callbacks is not None:
        options[ATTENTION_CALLBACKS_OPTION] = callbacks
    kwargs = {}
    if references:
        kwargs = {
            "ref_latents": [
                torch.zeros((batch, 2, height, width))
                for height, width in references
            ],
            "ref_latents_method": "index",
        }
    return adapter(
        torch.zeros((batch, 2, 2, 3)),
        torch.ones((batch,)),
        torch.zeros((batch, 5, 4)),
        transformer_options=options,
        **kwargs,
    )


class AttentionCallbackPlumbingTests(unittest.TestCase):
    def setUp(self):
        import nunchaku.models.transformers.transformer_flux2 as backend
        self.backend = backend
        self.old_version = getattr(backend, "FLUX2_ATTENTION_CALLBACK_API_VERSION", None)

    def tearDown(self):
        if self.old_version is None:
            self.backend.__dict__.pop("FLUX2_ATTENTION_CALLBACK_API_VERSION", None)
        else:
            self.backend.FLUX2_ATTENTION_CALLBACK_API_VERSION = self.old_version

    def test_no_callbacks_preserve_none_transport(self):
        adapter, transformer = adapter_and_transformer()
        run(adapter, batch=2)
        self.assertEqual(transformer.kwargs, [None, None])

    def test_active_callbacks_require_capability(self):
        adapter, _ = adapter_and_transformer()
        callbacks = Flux2AttentionCallbacks(pre_attention_callbacks=(lambda *args: None,))
        self.backend.__dict__.pop("FLUX2_ATTENTION_CALLBACK_API_VERSION", None)
        with self.assertRaisesRegex(RuntimeError, "callback API v1"):
            run(adapter, callbacks=callbacks)

    def test_serialized_batch_receives_ordered_layout_and_callbacks(self):
        adapter, transformer = adapter_and_transformer()
        pre = lambda *args: None
        post = lambda *args: None
        callbacks = Flux2AttentionCallbacks((pre,), (post,))
        self.backend.FLUX2_ATTENTION_CALLBACK_API_VERSION = 1
        run(adapter, batch=2, references=((1, 2), (2, 1)), callbacks=callbacks)
        self.assertEqual(len(transformer.kwargs), 2)
        for runtime in transformer.kwargs:
            self.assertEqual(runtime["pre_attention_callbacks"], (pre,))
            self.assertEqual(runtime["post_attention_callbacks"], (post,))
            self.assertEqual(runtime["generated_token_count"], 6)
            self.assertEqual(runtime["reference_token_counts"], (2, 2))
            self.assertEqual(runtime["reference_spatial_shapes"], ((1, 2), (2, 1)))
        self.assertIsNot(transformer.kwargs[0], transformer.kwargs[1])

    def test_callback_configuration_is_immutable_and_validated(self):
        with self.assertRaises(TypeError):
            Flux2AttentionCallbacks(pre_attention_callbacks=[])
        with self.assertRaises(TypeError):
            Flux2AttentionCallbacks(post_attention_callbacks=(object(),))

    def test_controller_requires_spatial_metadata_api_v2(self):
        adapter, _ = adapter_and_transformer()
        callback = lambda *args: None
        options = {
            ATTENTION_CALLBACKS_OPTION: Flux2AttentionCallbacks((callback,), ()),
            REF_LATENT_CONTROLLER_DIRECT_OPTION: (callback,),
        }
        self.backend.FLUX2_ATTENTION_CALLBACK_API_VERSION = 1
        with self.assertRaisesRegex(RuntimeError, "callback API v2"):
            adapter(
                torch.zeros((1, 2, 2, 3)),
                torch.ones((1,)),
                torch.zeros((1, 5, 4)),
                transformer_options=options,
            )


if __name__ == "__main__":
    unittest.main()
