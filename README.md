# ComfyUI-MultiGPU - DisTorch2 slim

A stripped fork of [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
that keeps **one node** and disables the rest.

## The node

`UNETLoaderDisTorch2MultiGPU` - "Load Diffusion Model (DisTorch2, multi-GPU)".
Splits a diffusion model across two GPUs at load time, using a donor card's VRAM as
overflow instead of streaming weights from system RAM on every step.

Inputs: `unet_name`, `weight_dtype`, `compute_device`, `virtual_vram_gb`,
`donor_device`, `expert_mode_allocations`, `eject_models`.

Sizing `virtual_vram_gb`: run once with `0` and read the log - DisTorch2 prints
`set 'virtual_vram_gb' to at least N`. Use N + 1..2 GB. More is slower: the split is
sequential, so extra donor moves layers onto the slower card.

## Measured

MiniMax H3 Image, 37.5 GB bf16, 4736x3552, RTX 5090 + RTX 4090, no P2P:

| setup | per step | 6 steps |
|---|---|---|
| single GPU, 9.5 GB offloaded to RAM | 25.7 s | 2:34 |
| DisTorch2, donor 14 GB | **7.05 s** | 0:42 |
| DisTorch2, donor 18 GB | 8.00 s | 0:48 |

Output identical to the single-GPU render.

## Changes vs upstream

* Registers only `UNETLoaderDisTorch2MultiGPU`.
* Disables the cross-device patch stack (comfy_kitchen DLPack guard, stream patch,
  sample device guard). Those move loaded int8 tensors between GPUs and raise
  `cudaErrorIllegalAddress` on cards without P2P.
* Imports `lowvram_available` defensively - ComfyUI 0.46 removed it, which broke
  DisTorch2 loading with `cannot import name`.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/wolfden/ComfyUI-MultiGPU
```

Not in Manager's catalog; install by Git URL.

MIT, as upstream.
