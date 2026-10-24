import torch
import triton
import triton.language as tl


# ============================================================
# 第1个 Kernel：向量相加  C = A + B
# ============================================================
# CUDA 对照:
#   @triton.jit          ↔  __global__
#   tl.program_id(0)      ↔  blockIdx.x
#   tl.arange(...)        ↔  threadIdx.x (自动展开)
#   tl.load / tl.store    ↔  全局内存读写
#   grid=(n_blocks,)      ↔  <<<n_blocks, 1>>> (block 维度)
# ============================================================

@triton.jit # 编译 Triton Kernel
def vector_add_kernel(
    A_ptr,          # 输入向量 A 的首地址 (pointer)
    B_ptr,          # 输入向量 B 的首地址
    C_ptr,          # 输出向量 C 的首地址
    N,              # 向量总长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素个数 (编译期常量)
):
    """
    每个 program 处理 BLOCK_SIZE 个连续元素。
    CUDA 对照: 每个 block 处理 BLOCK_SIZE 个元素，一个 block 只开 1 个线程也没事，
    因为 Triton 在 block 内部自动帮你做向量化循环。
    """
    # ---- Step 1: 确定当前 program 负责的元素范围 ----
    # tl.program_id(0)  ↔  CUDA 的 blockIdx.x
    pid = tl.program_id(axis=0)

    # 计算本 program 要处理的起始偏移量
    # 例如: pid=2, BLOCK_SIZE=1024 → offset = 2048
    offset = pid * BLOCK_SIZE

    # 生成一个索引数组: [offset, offset+1, offset+2, ..., offset+BLOCK_SIZE-1]
    # tl.arange(0, BLOCK_SIZE)  ↔  CUDA 里 for (int i = threadIdx.x; i < BLOCK_SIZE; i++)
    indices = offset + tl.arange(0, BLOCK_SIZE)

    # 生成一个 mask: 哪些索引是有效的 (防止越界)
    # 因为 N 不一定能被 BLOCK_SIZE 整除
    mask = indices < N

    # ---- Step 2: 从 HBM 加载数据到寄存器 ----
    # tl.load  ↔  CUDA 的全局内存读取
    # mask 参数: 只加载有效位置的元素，越界的会填充 0 (或用 other 指定填充值)
    a = tl.load(A_ptr + indices, mask=mask)
    b = tl.load(B_ptr + indices, mask=mask)

    # ---- Step 3: 计算 ----
    c = a + b

    # ---- Step 4: 写回 HBM ----
    # tl.store  ↔  CUDA 的全局内存写入
    tl.store(C_ptr + indices, c, mask=mask)


# ============================================================
# 包装函数: 方便从 Python 调用
# ============================================================
def vector_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """对 PyTorch Tensor 做 Triton 向量加法"""
    assert A.shape == B.shape, "A 和 B 的形状必须一致"
    assert A.is_cuda and B.is_cuda, "输入必须在 GPU 上"

    N = A.numel() # 向量总长度
    C = torch.empty_like(A)  # 分配输出 tensor

    # 选择 block size (每个 program 处理的元素数)
    BLOCK_SIZE = 1024

    # 计算需要多少个 program
    # grid = (n_blocks,)  ↔  CUDA 的 <<<n_blocks, 1>>>
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # 启动 kernel
    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)  # type: ignore

    return C


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 确保有 GPU
    assert torch.cuda.is_available(), "需要 CUDA GPU"

    # 构造随机测试数据
    N = 100_000
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)

    # Triton 实现
    C_triton = vector_add(A, B)

    # PyTorch 实现 (作为参考答案)
    C_torch = torch.add(A, B)

    # 对比结果
    diff = torch.abs(C_triton - C_torch)
    max_diff = diff.max().item()
    all_close = torch.allclose(C_triton, C_torch) # 检查所有元素是否在允许误差范围内

    print(f"向量长度: {N}")
    print(f"最大误差: {max_diff:.10f}")
    print(f"结果一致: {all_close}")

    if all_close:
        print("\n✅ 通过！Triton 向量加法结果与 torch.add() 完全一致")
    else:
        print(f"\n❌ 失败！存在差异")