# Nunchaku 1.4.1 backend installation and rollback

This document applies only to the runtime-qualified Windows candidate:

- Windows x86-64, CPython 3.13
- PyTorch 2.14.0+cu130 and CUDA 13.0
- `nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256 `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`

Do not install this wheel into the production-qualified PyTorch 2.11 environment.
Provision and keep a separate environment so rollback is a launcher/environment
switch rather than an attempt to reconstruct an overwritten installation.

## Prerequisites

1. A separately provisioned 64-bit CPython 3.13 environment.
2. PyTorch `2.14.0+cu130` and its matching torchvision build, installed using
   an approved source for that environment.
3. A CUDA-capable NVIDIA GPU. Runtime qualification used an RTX 3060 12 GB.
4. A compatible ComfyUI checkout. Qualification used commit
   `1e719279f56558b2c6d5d2f2aa590b197559f1f6` (`v0.45.0-40-g1e719279f`).
5. The custom node and the exact candidate wheel listed above.

The wheel metadata declares `accelerate>=1.10`, `diffusers>=0.41`, `einops`,
`huggingface-hub>=1.32,<3`, `peft>=0.20`, `protobuf`, `sentencepiece`,
`torchvision>=0.22`, and `transformers>=5.4`. It does not declare PyTorch, so
the exact PyTorch/CUDA build must be provisioned and verified separately.

## Side-by-side installation

In PowerShell, activate the new environment and verify it before installing the
backend:

```powershell
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); assert sys.version_info[:2] == (3, 13); assert torch.__version__ == '2.14.0+cu130'; assert torch.version.cuda == '13.0'"
```

Verify the candidate without renaming it:

```powershell
$wheel = 'C:\path\to\nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl'
$expected = 'AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF'
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $wheel).Hash
if ($actual -ne $expected) { throw "Candidate wheel SHA-256 mismatch: $actual" }
python -m pip install $wheel
python -m pip check
```

This command may resolve the wheel's declared Python dependencies in the new
environment. Review the resolver plan before accepting unexpected changes. Do
not run it in the retained PyTorch 2.11 environment.

Verify the installed identity, extension import, and callback API:

```powershell
python -c "import importlib.metadata as m, torch; import nunchaku._C; from nunchaku.models.transformers.flux2_attention_callbacks import FLUX2_ATTENTION_CALLBACK_API_VERSION as api; print(m.version('nunchaku'), torch.__version__, torch.version.cuda, api); assert m.version('nunchaku') == '1.4.1+cu13.0torch2.14.kleincallback2'; assert api == 3"
```

Start ComfyUI with this environment and confirm that the 4B/9B loader, LoRA,
Direct K/V, Enhancer, and Identity Feature Transfer nodes register without an
unsupported-backend warning. Run one minimal 4B workflow and one minimal 9B
workflow sequentially. Confirm finite decoded output and unload the first model
before loading the second.

## Rollback

The rollback unit is the complete retained environment, not only the Nunchaku
package.

1. Stop the ComfyUI process using the candidate environment.
2. Restore the launcher or service configuration to the retained qualified
   PyTorch 2.11 environment.
3. Start ComfyUI from that environment and verify:
   - Python 3.13;
   - PyTorch `2.11.0+cu130`;
   - Nunchaku `1.3.0.dev20260629+cu13.0torch2.11.kleincallback2`;
   - callback API v3;
   - one known 4B or 9B workflow.

The exact qualified baseline archive is:

- `nunchaku-1.3.0.dev20260629+cu13.0torch2.11.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256 `759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C`

Keep that archive and its working environment before evaluating 1.4.1. Do not
install the PyTorch 2.11 wheel into the PyTorch 2.14 environment as a substitute
for switching back to the retained baseline environment.

## Scope and troubleshooting

- Linux x86-64 and Linux aarch64 derivatives are archive-verified only; they
  are not runtime-qualified by this release.
- Other Python, PyTorch, CUDA, GPU, and operating-system combinations are
  untested unless explicitly listed in a later qualification report.
- A missing callback module or callback API other than v3 is an unsupported
  backend, not a condition to bypass.
- The runtime-qualified environment passed its declared dependency audit and
  `pip check`. Dependency resolution from a newly created empty environment was
  not qualified.
- The patched wheel retains the upstream Apache-2.0 license. Its upstream
  compiled extension is byte-identical to the authoritative upstream wheel,
  but a source-reproducible build of that binary has not been established.

