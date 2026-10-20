# Qwen-Image-2.1 on RTX 3060 12GB: VAE seam workaround and BF16 partial GPU residency

Author: Makoto Watanabe (GitHub: [nabe2030](https://github.com/nabe2030))  
Experiment date: October 19, 2026

## Overview

We evaluated two changes using one RTX 3060 12GB and the Qwen-Image-2.1 derivative **UC-BF16**:

1. **A workaround for purple seams in tiled VAE decoding.** Excluding 16px at internal tile edges and blending with shared RGBA weights and FP32 normalization removed the visible seam in one visually inspected test image.
2. **Partial GPU residency for the Transformer.** Keeping 16 of its 32 BF16 blocks on the GPU reduced repeated CPU-to-GPU weight transfers. In a single comparison using the same profiler settings, denoise time decreased from **143.783 s to 107.846 s (approximately 25%)**, with a bitwise-identical output latent.

A combined run without the profiler completed **1024×1024, 24-step generation through image saving in 123.976 s**.

## Test environment

| Item | Configuration |
| --- | --- |
| Image-generation GPU | One NVIDIA RTX 3060 12GB |
| PCIe connection under load | Gen4 x4 |
| Checkpoint | Qwen-Image-2.1 UC-BF16 derivative |
| Weight format | Unquantized BF16 safetensors |
| Transformer | Approximately 7.115B parameters, 32 blocks, 13.253 GiB of BF16 weights |
| Generation | 1024×1024, 24 steps, CFG 1 |
| Reference images | None |
| Prefix KV cache | Enabled |
| Baseline memory management | Sequential CPU offload at submodule granularity |
| VAE | AutoencoderKLQwenImage21, BF16 |
| VAE tiles | 256px tiles at 192px stride |

A Gemma workload on a separate GPU remained running during the experiments.

These experiments used a derivative checkpoint; no comparison with the official checkpoint was performed. Exact CUDA-environment package versions and code revisions have not yet been compiled for this public report.

## 1. VAE seam workaround: same-latent comparison

The denoiser ran once. The resulting latent was decoded using two methods, both with the same 36-tile layout:

- **A: Standard decoding.** Standard blending, cropping, and concatenation.
- **B: Workaround.** Exclude 16 output pixels at internal tile borders and feather-blend the remaining 32px overlaps. Apply common weights to RGBA, with FP32 accumulation and normalization. Preserve physical image borders.

Neither forced opacity nor post-export PNG processing was used.

| Metric | Standard A | Workaround B |
| --- | ---: | ---: |
| Decode wall time, including CPU offload | 7.484 s | 8.219 s |
| Peak PyTorch allocated memory | 580.64 MiB | 462.09 MiB |
| Peak PyTorch reserved memory | 686 MiB | 530 MiB |
| Mean RGB boundary difference, normalized 0–1 | 0.007755 | 0.007352 |
| Mean alpha boundary difference, normalized 0–1 | 0.0008681 | 0.0001573 |
| Maximum alpha boundary difference | 0.12793 | 0.003906 |
| Minimum alpha across the image | 0.71777 | 0.95898 |

A reproduced the original image pixel-for-pixel. With B, **the author visually confirmed that the purple seam disappeared in the test image**.

Mean alpha boundary difference decreased by approximately 82%, while decode time increased by approximately 9.8%.

Boundary metrics also include genuine image-content edges. RGB and purple-component metrics did not improve uniformly. **This is a workaround visually verified on one image; it does not establish equivalence to untiled decoding or guarantee removal on all images.**

B also decoded saved latents from different seeds at the following sizes without OOM. Visual assessment of these three additional images remains incomplete.

| Size | B decode time |
| --- | ---: |
| 1152×1152 | 9.020 s |
| 1376×1024 | 11.106 s |
| 1024×1376 | 11.039 s |

## 2. Investigating generation latency

We traced one baseline 24-step denoise with Nsight Systems and NVTX.

| Recorded interval category | Wall time | Fraction |
| --- | ---: | ---: |
| GPU kernels only | 68.636 s | 47.74% |
| Memcpy only | 57.993 s | 40.33% |
| Kernel/Memcpy overlap | 0.000 s | 0% |
| Neither kernel nor Memcpy recorded | 17.155 s | 11.93% |
| Total denoise interval | 143.783 s | 100% |

CPU-to-GPU transfer totaled approximately 341.533 GB, matching 24 passes of the Transformer weights. Throughput during active HtoD transfer intervals was approximately 5.889 GB/s.

CPU backing weights were retained; this configuration did not copy all weights back from GPU to CPU after each operation.

In the captured trace, **weight transfers and GPU work were predominantly serial**. Nsight warned of potentially missing events, so the zero-overlap result is an observation from the captured trace.

## 3. Keep 16 Transformer blocks on the GPU

The complete Transformer does not fit within 12GB of VRAM, so we kept 16 of its 32 blocks resident.

The budget reserved 1,420 MiB for the dynamic region and another 2,560 MiB of safety allowance. Resident BF16 weight payload was approximately 6,656 MiB.

| Comparison using the same Nsight settings | Sequential baseline | 16 blocks resident |
| --- | ---: | ---: |
| Denoise | 143.783 s | 107.846 s |
| GPU kernel time | 68.636 s | 68.609 s |
| Memcpy time | 57.993 s | 29.092 s |
| Other intervals | 17.155 s | 10.145 s |
| Peak allocated memory | 760.63 MiB | 7,416.65 MiB |
| Peak reserved memory | 1,420 MiB | 7,602 MiB |
| CPU-to-GPU transfer | 341.533 GB | 174.029 GB |

Residency preparation took 1.146 s and was measured separately from denoise.

This single comparison showed:

- Approximately **25% less denoise time**.
- Approximately **49% less CPU-to-GPU transfer**.
- No recorded weight HtoD events for the resident 16 blocks across all 24 steps.
- **Bitwise equality** with all 262,144 elements of the baseline latent.
- Maximum difference, mean difference, and RMSE all equal to zero.
- No quantization, model change, or OOM.

GPU kernel time was almost unchanged; reduced transfers and associated processing account for the observed time reduction.

Each configuration was measured once. Profiler overhead, temperature/clock variation, and possible missing events limit generalization. The observed reduction is not a general performance guarantee.

## 4. Combine partial residency with the VAE workaround

We combined 16 resident blocks and VAE B for one generation without Nsight.

| Stage | Wall time |
| --- | ---: |
| Residency preparation | 1.208 s |
| Text conditioning | 3.382 s |
| Denoise | 107.095 s |
| VAE B decode | 8.108 s |
| Image postprocessing | 0.024 s |
| PNG encoding | 0.309 s |
| **Request start through completed image save** | **123.976 s** |

Model loading was also measured separately at 2.912 s. The request-to-save total includes loading, state restoration, LAN communication, and save checks. Do not add those individual stages to the total again.

The generated latent was bitwise identical to the baseline. The RGBA image also matched the previously visually accepted B image pixel-for-pixel.

Peak allocated/reserved memory was 7,416.65/7,602 MiB, with no OOM.

This unprofiled result cannot be compared directly with the profiled baseline as an exact speed ratio.

## 5. Testbed integration

The tested combination was added as an explicitly selectable experimental option in the Testbed UI.

Its scope is limited to:

- UC-BF16.
- 1024×1024.
- 24 steps.
- No reference images.
- CFG 1.
- Prefix KV cache enabled.

Standard behavior remains the default. Unsupported conditions use the original path, and selection, actual application, and reason are recorded in history.

After success, cancellation, or an exception, the worker restores standard behavior. Existing images/history and the separate-GPU Gemma workload were retained.

Checks after UI integration consisted of isolated tests and browser checks. **No additional real GPU generation was performed through the newly integrated route; this is distinct from the preceding generation experiment.**

## Remaining work

- Repeated measurements to quantify timing variation.
- Image quality and VRAM use across other seeds and sizes.
- Reference-image editing.
- CUDA package-version and code-revision inventory.
- A reproducible procedure using publishable inputs.

## Related discussions

- [CUDA experiment report in the Qwen official issue](https://github.com/QwenLM/Qwen-Image-2.1/issues/5#issuecomment-6279330010)
- [Earlier VAE issue report on ROCm](https://github.com/huggingface/diffusers/issues/14928)
- [vLLM-Omni proposal using a different tile-layout correction](https://github.com/vllm-project/vllm-omni/pull/8868)

Implementation and measurement analysis used AI assistance. The author performed the stated visual assessment.
