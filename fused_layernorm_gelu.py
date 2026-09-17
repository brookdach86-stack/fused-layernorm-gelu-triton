import torch
import triton
import triton.language as tl

@triton.jit
def fused_layernorm_gelu_kernel(
  X_ptr, Y_ptr,
  Gamma_ptr, Beta_ptr,
  N,
  stride_row,
  eps: tl.constexpr,
  BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis = 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + row_idx * stride_row + offsets, mask = mask, other = 0.0)
    mean = tl.sum(x, axis = 0) / N
    var = tl.sum((x - mean) * (x - mean), axis = 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    gamma = tl.load(Gamma_ptr + offsets, mask = mask, other = 1.0)
    beta = tl.load(Beta_ptr + offsets, mask = mask, other = 0.0)
    y = gamma * x_hat + beta
    # GELU activation
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    tl.store(Y_ptr + row_idx * stride_row + offsets, y, mask = mask)


def fused_layernorm_gelu(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda
    assert x.dim() == 2

    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    fused_layernorm_gelu_kernel[grid](
        x, y, gamma, beta,
        n_cols, x.stride(0), eps,
        BLOCK_SIZE = BLOCK_SIZE,
    )
    return y

if __name__ == '__main__':
    assert torch.cuda.is_available()
    x = torch.randn(128, 512, device='cuda')
    gamma = torch.ones(512, device='cuda')
    beta = torch.zeros(512, device='cuda')
    y_triton = fused_layernorm_gelu(x, gamma, beta)
    y_torch = torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, [512], gamma, beta))
    max_diff = torch.abs(y_triton - y_torch).max()
    all_close = torch.allclose(y_triton, y_torch, atol=1e-5)
    print(f"最大误差: {max_diff}")
    print(f"结果一致: {all_close}")
