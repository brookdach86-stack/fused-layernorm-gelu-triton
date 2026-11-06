# Fused LayerNorm + GELU in Triton

Triton kernel 演进链：向量加 → 矩阵乘 → SoftMax → LayerNorm + GELU 融合。

## 环境

RTX 5060 Laptop（sm_120），Python 3.10+，PyTorch + Triton。

## 演进链

| 阶段 | 文件 | 干了什么 | 新概念 |
| --- | --- | --- | --- |
| 01 | `01_vector_add/vector_add.py` | 第一个 Triton kernel：向量相加 | `@triton.jit`、`program_id`、`tl.load/store` |
| 02 | `02_matmul/matmul.py` | 矩阵乘法，block pointer 自动做 shared memory tiling | `tl.dot()`（调 Tensor Core）、分块 |
| 03 | `03_softmax/softmax.py` | SoftMax，online 算法一趟扫完 | 归约 `tl.max/sum`、数值稳定 |
| 04 | `04_fused_layernorm_gelu/` | LayerNorm + GELU 融合，中间结果不落显存 | 算子融合、带宽优化 |

## 04 LayerNorm + GELU 融合

把 LayerNorm 和 GELU 融合成一个 kernel，中间结果不落显存，直接在寄存器里传给下一步。

跑 `benchmark.py` 和 PyTorch 分步实现对比，融合版本快 **2.1 倍**，显存省 **33.29%**（N=4096, M=512）。

### 文件

- `layernorm.py` — 单独的 LayerNorm kernel，和 `torch.nn.functional.layer_norm` 结果一致
- `fused_layernorm_gelu.py` — 融合 kernel，LayerNorm 输出直接接 GELU
- `benchmark.py` — 用 `triton.testing.perf_report` 跑不同输入长度下的耗时对比，另外测了显存占用

### 为什么快

GPU 上这种逐元素算子基本是显存带宽瓶颈，不是算力瓶颈。分开算的话 LayerNorm 输出要写一次显存、GELU 再读一次，融合之后省掉这一次读写。

### 踩过的坑

- 方差在浮点精度下偶尔会出现微小负数，加了 `tl.where(var < 0, 0, var)` 兜底
- `BLOCK_SIZE` 必须是 2 的幂，用 `triton.next_power_of_2(N)` 算
- GELU 用的是精确公式（`tl.erf`），没用 tanh 近似
- 显存用 `torch.cuda.max_memory_allocated()` 测峰值，不能用 `memory_allocated()`，后者只是当前值

## Triton 和 CUDA 的区别

CUDA 从 thread 写起，管 shared memory、bank conflict、寄存器溢出，改一行代码要琢磨半天。Triton 从 block 写起，编译器帮你拆成 thread——研究的时候快太多了，先跑通再说。

复杂算子（FlashAttention、GEMM）的最终性能还是得靠 CUDA 手写调优，Triton 适合快速验证算法和写融合 kernel。
