# LayerNorm + GELU 融合算子

用 Triton 把 LayerNorm 和 GELU 融合成一个 kernel,中间结果不落显存,直接在寄存器里传给下一步。

跑 benchmark.py 和 PyTorch 分步实现对比,融合版本快 2.1 倍,显存省 33.29%（N=4096,M=512）。

## 文件

- `layernorm.py` — 单独的 LayerNorm kernel,和 `torch.nn.functional.layer_norm` 结果一致
- `fused_layernorm_gelu.py` — 融合 kernel,LayerNorm 输出直接接 GELU
- `benchmark.py` — 用 `triton.testing.perf_report` 跑不同输入长度下的耗时对比,另外测了显存占用

## 为什么快

GPU 上这种逐元素算子基本是显存带宽瓶颈,不是算力瓶颈。分开算的话 LayerNorm 输出要写一次显存、GELU 再读一次,融合之后省掉这一次读写。

## 踩过的坑

- 方差在浮点精度下偶尔会出现微小负数,加了 `tl.where(var < 0, 0, var)` 兜底
- `BLOCK_SIZE` 必须是 2 的幂,用 `triton.next_power_of_2(N)` 算
- GELU 用的是精确公式(`tl.erf`),没用 tanh 近似
- 显存用 `torch.cuda.max_memory_allocated()` 测峰值,不能用 `memory_allocated()`,后者只是当前值
