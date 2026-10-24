import torch
import triton
import triton.language as tl

# ============================================================
# Softmax:  y_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
# 新概念: tl.exp(), tl.max(), tl.sum() — 归约操作
# 数值稳定: 减去 max 防止 exp 溢出
# 其实就是每个元素占“总指数和”的多少比例
# ============================================================

@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    N,              # 每行长度
    stride_row,     # 跳一行跨多少元素
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 处理第几行
    row_idx = tl.program_id(axis=0)
    row_start = X_ptr + row_idx * stride_row

    # 当前行所有元素的偏移
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Step 1: 加载一行数据
    x = tl.load(row_start + offsets, mask=mask, other=float("-inf"))

    # Step 2: 求 max (数值稳定关键: 减去 max 防止 exp 溢出)
    x_max = tl.max(x, axis=0)

    # Step 3: 计算 exp(x - max)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)

    # Step 4: 求和
    sum_exp = tl.sum(exp_x, axis=0)

    # Step 5: 归一化
    y = exp_x / sum_exp

    # Step 6: 写回
    y_ptr = Y_ptr + row_idx * stride_row
    tl.store(y_ptr + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "输入必须在 GPU 上"
    assert x.dim() == 2, "只支持 2D 输入 (batch, dim)"

    y = torch.empty_like(x)
    n_rows, n_cols = x.shape

    # 每个 program 处理的元素数量，必须是 2 的幂次方
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)  # 每行一个 program

    softmax_kernel[grid](
        x, y, n_cols,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    x = torch.randn(128, 512, device="cuda", dtype=torch.float32)

    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=1)

    max_diff = torch.abs(y_triton - y_torch).max().item()
    all_close = torch.allclose(y_triton, y_torch, atol=1e-5)

    print(f"输入大小: {x.shape}")
    print(f"最大误差: {max_diff:.10f}")
    print(f"结果一致: {all_close}")
    print("✅ 通过！" if all_close else "❌ 失败！")