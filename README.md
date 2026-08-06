# ComfyUI-Nunchaku-Klein

ComfyUI custom node for **Nunchaku Flux.2 Klein**, independent of official [ComyUI-Nunchaku](https://github.com/nunchaku-ai/nunchaku).
Added features such as **Klein Enhancer** ported to work on Nunchaku.

> **NOTE**: some of the Enhancer nodes are not implemented yet. They appear in node list, but they won't work.
> 
> What's supported now:
> - Text Enhancer
> - Enhancer
> 
> What's NOT supported:
> - All the other Enhancer related nodes

## News
- **2026-08-05** **v0.2.0**
  - Klein Enhancer node. (experimental)
- **2026-08-04**
  - Foundation and basic functionalities for **9B** have been mostly complete.
  - **LoRA** support
  - **Multi-reference editing** support
  - First experimental implementation of **Klein Text Enhancer** node.
    - **NOTE**: Our "Enhancer" nodes **do not work exactly the same as the [original Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)**, but the behavior is close. Still experimental.
  - **4B not supported yet**.

## Todos:

- [x] 9B basic support
- [x] LoRA support
- [x] Reference Edit support
- [x] Klein Enhancer (partially)
- [ ] Test noise mask and Differential Diffusion
- [ ] Test Normalized Attention Guidance
- [ ] 4B support

## Installation

1. **First you need to install [tonera's fork of nunchaku (Vitoom Nunchaku)](https://huggingface.co/tonera/vitoom-nunchaku)**. Choose the pre-built wheel matching your setup from their repository.
2. Clone this repository.
3. Download tonera's Nunchaku Klein model from their repository if you don't have one.

**Tested with ComfyUI v0.29**. Older versions may not work.

## Related Projects
This project involves/is related to several third-party projects

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [Official Nunchaku](https://github.com/nunchaku-ai/nunchaku)
- [Community-fork of Nunchaku backend](https://huggingface.co/tonera/vitoom-nunchaku)
- [ComfyUI-Flux2Klein-Enhancer](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer)

## License
GPL-3.0-or-later