# ComfyUI-Nunchaku-Klein

![](docs/media/screenshot-2026-08-11-200143.jpg)

ComfyUI custom node for **Nunchaku Flux.2 Klein**, independent of official [ComyUI-Nunchaku](https://github.com/nunchaku-ai/nunchaku).
Added features such as **Klein Enhancer** ported + compatibility version to work on Nunchaku (not identical).

> **NOTE**: In order to use "**Direct K/V**" nodes or **Identity Feature Transfer**, you'll have to install the newer fork of Vitoom Nunchaku.

## News
- **2026-08-11**
  - **v0.8.0**
    - **Text/Ref Balance (Direct K/V)** (needs the new modified backend)
  - **v0.7.0**
    - **Ref Latent Weight (Direct K/V)**
      - **NOTE:** NEEDS NEW NUNCHAKU BACKEND. To use this node you'll need to reinstall nunchaku with our **"fork of the community-maintained Nunchaku"** that means, _Vitoom Nunchaku_ which is an extended version of nunchaku by _tonera_ has been **further extended** by me, which gives you **full control over Flux.2 Klein**.
- **2026-08-09**
  - **v0.6.0**
    - **Text/Ref Balance** (not perfect, experimental. Use `0.50-1.00` in `balance`.)
  - **v0.5.0**
    - **Ref Latent Weight**
- **2026-08-08** **v0.4.1**
  - **Multi Reference Latent**
  - **Mask Ref Controller**
  - **Color Anchor**
- **2026-08-06** **v0.3.0**
  - **Sectioned Encoder**
  - **Detail Controller**
- **2026-08-05** **v0.2.0**
  - **Klein Enhancer**
- **2026-08-04**
  - Foundation and basic functionalities for **9B** have been mostly complete.
  - **LoRA** support
  - **Multi-reference editing** support
  - First experimental implementation of **Klein Text Enhancer** node.
    - **NOTE**: Our "Enhancer" nodes **do not work exactly the same as the [original Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)**, but the behavior is close. Still experimental.
  - **4B not supported yet**.

**Observed**: repeated Nunchaku FLUX.2 Klein executions can produce materially different outputs despite identical workflow inputs and seed. This also reproduces in plain T2I without references, Ref Latent Weight, or LoRA, so it is not specific to those target features.

## Todos:

- [x] 9B basic support
- [x] LoRA support
- [x] Reference Edit support
- [x] Klein Enhancer (partially)
- [ ] Test Differential Diffusion / if not: SpotEdit
- [ ] ~~Test Normalized Attention Guidance~~
- [ ] 4B support

## Installation

### Basic Usage

1. **First you need to install [tonera's fork of nunchaku (Vitoom Nunchaku)](https://huggingface.co/tonera/vitoom-nunchaku)**. Choose the pre-built wheel matching your setup from their repository.
2. Clone this repository.
3. Download tonera's Nunchaku Klein model from their repository if you don't have one.

### Direct K/V

If you want to use Ref Latent Weight (Direct K/V), Text/Ref Balance (Direct K/V) or Identity Feature Transfer:

Either:
- Install [my "fork of the fork" of nunchaku](https://github.com/tom-m-2020/vitoom-nunchaku-extended).
- Use "repack" script in the repo to your chosen pre-built wheel (**if the version exactly match**.)
  - If you use the repack script, they MUST match the exact variant of pre-built _Vitoom Nunchaku_ wheel as the source e.g.:
    - If you have `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64`, you must use the script that targets `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64`.

#### Currently supported wheels

Only a limited number of versions are supported now.

- `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64`

(Basic usage supports all the other versions, too.)

### Environment

**Tested with**:
- **ComfyUI >=0.29**. Older versions may not work.
- **Python 3.13**
- **Torch 2.11**
- **CUDA 13.0**

## Related Projects
This project involves/is related to several third-party projects

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [Official Nunchaku](https://github.com/nunchaku-ai/nunchaku)
- [Community-fork of Nunchaku backend](https://huggingface.co/tonera/vitoom-nunchaku)
- [ComfyUI-Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)

## License
GPL-3.0-or-later