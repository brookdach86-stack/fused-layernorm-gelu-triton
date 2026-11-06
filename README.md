# Fused LayerNorm + GELU in Triton

Triton kernel evolution chain: vector add → matmul → SoftMax → fused LayerNorm + GELU.

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
