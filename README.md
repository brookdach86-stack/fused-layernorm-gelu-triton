![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120)-brightgreen)

---

<!--
@Author: Wain (brookdach86-stack@users.noreply.github.com)
@GitHub: https://github.com/brookdach86-stack
@Note: All original work, for citation please include source. Learning path: CSAPP Lab → CUDA Kernel → Triton → LLM Inference
-->

# Fused LayerNorm + GELU in Triton

Triton kernel evolution chain — the point where I stopped treating each op as separate and started fusing. Key insight: LayerNorm output → GELU input is a register-level pass, no HBM round trip needed.

**Pitfalls I hit**:

- Variance goes microscopically negative in floating-point precision — added `tl.where(var < 0, 0, var)` guard
- `BLOCK_SIZE` must be power-of-two; use `triton.next_power_of_2(N)` to compute, otherwise silent wrong results
- GELU uses exact `tl.erf` not tanh approximation; approximation changes model accuracy
- Peak memory must use `torch.cuda.max_memory_allocated()` not `memory_allocated()` (latter is current, not peak)

## Environment

RTX 5060 Laptop (sm_120), Python 3.10+, PyTorch + Triton.

## Evolution Chain

| Stage | File | What It Does | New Concept |
| --- | --- | --- | --- |
| 01 | `01_vector_add/vector_add.py` | First Triton kernel: vector addition | `@triton.jit`, `program_id`, `tl.load/store` |
| 02 | `02_matmul/matmul.py` | Matrix multiplication, block pointers auto-generate shared memory tiling | `tl.dot()` (dispatches Tensor Core), tiling |
| 03 | `03_softmax/softmax.py` | SoftMax, online algorithm in single pass | Reduction `tl.max/sum`, numerical stability |
| 04 | `04_fused_layernorm_gelu/` | LayerNorm + GELU fusion, intermediate result never hits HBM | Operator fusion, bandwidth optimization |

## 04 LayerNorm + GELU Fusion

Fuse LayerNorm and GELU into one kernel — intermediate result stays in registers, passed directly to the next step.

Run `benchmark.py` to compare against PyTorch stepwise implementation. Fused version is **2.1× faster** and saves **33.29% memory** (N=4096, M=512).

### Files

- `layernorm.py` — Standalone LayerNorm kernel, matches `torch.nn.functional.layer_norm`
- `fused_layernorm_gelu.py` — Fused kernel, LayerNorm output feeds directly into GELU
- `benchmark.py` — Uses `triton.testing.perf_report` for latency comparison at various input lengths; also measures peak memory

### Why It's Fast

Element-wise operators like this on GPU are bandwidth-bound, not compute-bound. Split execution writes LayerNorm output to HBM once, then GELU reads it back. Fusion eliminates this round trip.

### Bugs Hit Along the Way

- Variance occasionally goes microscopically negative in floating-point precision; added `tl.where(var < 0, 0, var)` guard
- `BLOCK_SIZE` must be power of two; use `triton.next_power_of_2(N)` to compute
- GELU uses the exact formula (`tl.erf`), not tanh approximation
- Peak memory measured with `torch.cuda.max_memory_allocated()`, not `memory_allocated()` (the latter is current, not peak)

## Triton vs CUDA

CUDA is written from thread up — manage shared memory, bank conflicts, register spilling; changing one line takes careful thought. Triton is written from block up — the compiler decomposes it to threads — so iteration is much faster when prototyping algorithms.

For final performance of complex operators (FlashAttention, GEMM), handwritten CUDA is still needed. Triton excels at quickly validating algorithms and writing fused kernels.
