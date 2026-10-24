import torch
import triton
import triton.language as tl

# ============================================================
# 矩阵乘法  C[M,N] = A[M,K] @ B[K,N]
# 核心思想: 分块 (Tiling) — 把大矩阵切成小块，沿 K 维度循环累加
# 新概念: tl.dot() ↔ CUDA 的 Tensor Core WMMA/MMA 指令
# ============================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A 的行/列步长
    stride_bk, stride_bn,   # B 的行/列步长
    stride_cm, stride_cn,   # C 的行/列步长
    BLOCK_M: tl.constexpr,  # C 的行块大小
    BLOCK_N: tl.constexpr,  # C 的列块大小
    BLOCK_K: tl.constexpr,  # K 维度的分块大小
):
    # Step 1: 当前 program 负责 C 的哪一块 (2D grid)
    pid_m = tl.program_id(axis=0)   # ↔ blockIdx.x
    pid_n = tl.program_id(axis=1)   # ↔ blockIdx.y
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Step 2: 当前块的行列索引
    rows = off_m + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    cols = off_n + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # Step 3: 累加器初始化为零
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Step 4: 沿 K 维度分块循环累加
    # C_block = Σ A_block @ B_block
    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)   # (BLOCK_K,)

    # 注：[:, None] 把行变列，[None, :] 把列变行，相加后自动展开成二维网格

        # 加载 A 的一块 [BLOCK_M, BLOCK_K]
        # 算地址->掩码->加载
        a_ptrs = A_ptr + rows[:, None] * stride_am + k_off[None, :] * stride_ak
        a_mask = (rows[:, None] < M) & (k_off[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B 的一块 [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_off[:, None] * stride_bk + cols[None, :] * stride_bn
        b_mask = (k_off[:, None] < K) & (cols[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法 + 累加 (调用 Tensor Core!)
        accumulator += tl.dot(a, b)

    # Step 5: 写回 C
    c_ptrs = C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "A 的列数必须等于 B 的行数"
    assert A.is_cuda and B.is_cuda, "输入必须在 GPU 上"

    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    M, K, N = 128, 96, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C_triton = matmul(A, B)
    C_torch = torch.matmul(A, B)

    max_diff = torch.abs(C_triton.float() - C_torch.float()).max().item() # 一维最大误差

    all_close = torch.allclose(C_triton.float(), C_torch.float(), atol=1e-2) # 是否一致

    print(f"矩阵大小: A[{M},{K}] @ B[{K},{N}] = C[{M},{N}]")
    print(f"最大误差: {max_diff:.6f}")
    print(f"结果一致 (atol=1e-2): {all_close}")
    print("✅ 通过！" if all_close else "❌ 失败！")