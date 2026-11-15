![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM%2C32MB%20L2)-brightgreen)
![Triton](https://img.shields.io/badge/Framework-Triton%20%7C%20PyTorch-orange)
![Kernel Fusion](https://img.shields.io/badge/Fusion-LayerNorm%20%2B%20GELU-9cf)

---

<!--
@Author: Wain (brookdach86-stack@users.noreply.github.com)
@GitHub: https://github.com/brookdach86-stack
@Note: All original work, for citation please include source. Learning path: CSAPP Lab → CUDA Kernel → Triton → LLM Inference
-->

# Fused LayerNorm + GELU in Triton

Triton kernel迭代链——从这里开始不再把每个op当成独立的东西，而是开始融合。核心洞察：LayerNorm的输出直接喂给GELU，数据全程停在register里，根本不需要写HBM。

踩过的坑：

- variance在fp精度下会变成微负数——加了`tl.where(var < 0, 0, var)`兜底
- `BLOCK_SIZE`必须是2的幂——用`triton.next_power_of_2(N)`算，否则结果静默错误
- GELU必须用精确公式（`tl.erf`），不能用tanh近似——近似会改变模型精度
- 测peak memory要用`torch.cuda.max_memory_allocated()`，不是`memory_allocated()`（后者是当前值不是峰值）

## Environment

RTX 5060 Laptop (sm_120), Python 3.10+, PyTorch + Triton。

## 迭代链

| 阶段 | 文件 | 做了什么 | 新概念 |
| --- | --- | --- | --- |
| 01 | `01_vector_add/vector_add.py` | 第一个Triton kernel：向量加法 | `@triton.jit`、`program_id`、`tl.load/store` |
| 02 | `02_matmul/matmul.py` | 矩阵乘，block pointer自动做smem tiling | `tl.dot()`（底层调Tensor Core）、tiling |
| 03 | `03_softmax/softmax.py` | SoftMax，online算法单趟完成 | reduction `tl.max/sum`、数值稳定性 |
| 04 | `04_fused_layernorm_gelu/` | LayerNorm + GELU融合，中间值不写HBM | operator fusion、带宽优化 |

## 04 LayerNorm + GELU 融合

LayerNorm和GELU合成一个kernel——LayerNorm的输出直接在register里喂给GELU，不写HBM。

跑`benchmark.py`和PyTorch分开算的版本对比。融合版**快2.1倍**、**省33.29%显存**（N=4096, M=512）。

### 文件

- `layernorm.py` — 独立LayerNorm kernel，和`torch.nn.functional.layer_norm`结果一致
- `fused_layernorm_gelu.py` — 融合kernel，LayerNorm输出直接喂GELU
- `benchmark.py` — 用`triton.testing.perf_report`测不同输入长度的latency，也测peak memory

### 为什么快

这类element-wise算子在GPU上是**带宽受限**的，不是算力受限。分开算时LayerNorm的输出先写HBM，GELU再读回来。融合把这个HBM round trip干掉了。

## Triton vs CUDA

CUDA是从thread往上写的——要自己管smem、bank conflict、register spilling；改一行代码要想清楚连锁反应。Triton是从block往上写的——编译器帮你拆成thread，所以算法原型迭代比CUDA快得多。

但复杂算子（FlashAttention、GEMM）要榨到极限性能还是得手写CUDA。Triton的强项是快速验证算法和写融合kernel。
