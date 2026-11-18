# Draft release notes: Nunchaku 1.4.1 callback backend

This proposed backend update carries the maintained FLUX.2 Klein callback API
v3 patch onto Nunchaku 1.4.1 for the qualified Windows/PyTorch 2.14 profile.
It preserves the project's 4B and 9B loaders, signed and composed LoRAs,
reference editing, Direct K/V controls, Enhancers, and Identity Feature
Transfer contracts.

## Qualified artifact

- `nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256 `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`
- Windows x86-64, Python 3.13, PyTorch 2.14.0+cu130, CUDA 13.0
- Runtime hardware: NVIDIA RTX 3060 12 GB

The current Nunchaku 1.3.0.dev20260629/PyTorch 2.11 backend remains the
production-qualified rollback baseline until an explicit promotion decision.

## Qualification summary

Gate 1, Gate 2A, Gate 2B, and Gate 3 passed their required Windows execution
contracts. Final acceptance additionally passed six native ComfyUI workflows:
4B and 9B enhanced T2I, mixed-signed-LoRA single-reference Direct K/V, and
two-reference Identity Feature Transfer. The 4B T2I case ran at 768×512; all
other final cases ran at 512×512 with eight sampling steps.

The candidate is installed in a separate PyTorch 2.14 environment. See
`BACKEND_1.4.1_INSTALLATION.md` for checksum verification, startup checks, and
environment-level rollback.

## Provenance and licensing

The authoritative upstream input is
`nunchaku-1.4.1+cu13.0torch2.14-cp313-cp313-win_amd64.whl`, SHA-256
`4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F`.
The maintained repacker makes Python-only FLUX.2 callback changes, updates
wheel metadata and RECORD, and preserves the compiled extension byte-for-byte
(SHA-256 `997178E9442170A4F68F2D6ED1736AE735CFE8A371885038558C377A368D7927`).
The upstream Apache-2.0 license and attribution file remain in the wheel.

The derivative archive is deterministically reproducible from the pinned
upstream wheel. This is not a claim that the upstream compiled extension is
reproducibly buildable from source; complete corresponding build provenance
was not established.

## Known limitations

- Runtime qualification applies only to the exact Windows wheel and tested
  environment. Linux x86-64 and aarch64 have archive-level evidence only.
- A new-empty-environment resolver transaction was not run. The provisioned
  qualified environment satisfies all declared requirements and passes
  `pip check`.
- Decoded images are diagnostic evidence. Controlled prompt-level perceptual
  equivalence across backend versions was not established, and fused-kernel
  executions are not expected to be bit exact.
- The observed VAE decoder CUDA residual is backend-independent and stabilized;
  it is not attributed to Nunchaku model residency.
- This release does not claim that FLUX.3, Qwen-Image 2.1, or other unrelated
  Nunchaku model families are implemented or runtime-qualified by this custom
  node.

