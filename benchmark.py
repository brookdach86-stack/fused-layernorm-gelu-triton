import torch
import triton
import triton.language as tl
from layernorm import layernorm
from fused_layernorm_gelu import fused_layernorm_gelu

# Benchmark：性能对比 + 画折线图 + 测显存
# 步骤：导入两个kernel → 用perf_report装饰器 → 运行出图 → 测显存节省

# triton.testing.pref_report 用于记录性能指标
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], # 输入维度
        x_vals=[512, 1024, 2048, 4096], # 输入维度取值
        line_arg='provider', # 实验组
        line_vals=['triton_fused', 'triton_separate', 'pytorch'], # 实验组取值
        line_names=['Triton融合', 'Triton分步', 'PyTorch分步'], # 实验组名称
        plot_name='fused-layernorm-gelu', # 图名称
        args={'M': 512}, # 其他参数
    )
)
def benchmark(M,N,provider):
    x = torch.randn(M,N, device='cuda')
    gamma = torch.ones(N, device='cuda')
    beta = torch.zeros(N, device='cuda')

    # Triton融合
    if provider == 'triton_fused':
        fn = lambda : fused_layernorm_gelu(x, gamma, beta)
    # Triton分步
    elif provider == 'triton_separate':
        fn = lambda : torch.nn.functional.gelu(layernorm(x, gamma, beta))
    # PyTorch分步
    elif provider == 'pytorch':
        fn = lambda : torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, [N], gamma, beta))

    # 计时，返回性能指标
    ms = triton.testing.do_bench(fn)
    return ms

def measure_memory():
    M, N = 512, 4096
    x = torch.randn(M, N, device='cuda')
    gamma = torch.ones(N, device='cuda')
    beta = torch.zeros(N, device='cuda')

    torch.cuda.reset_peak_memory_stats()
    fused_layernorm_gelu(x, gamma, beta)
    mem_fused = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, [N], gamma, beta))
    mem_pytorch = torch.cuda.max_memory_allocated()

    save_pct = (mem_pytorch - mem_fused) / mem_pytorch * 100
    print(f"融合算子显存: {mem_fused / 1024**2:.2f} MB")
    print(f"PyTorch分步显存: {mem_pytorch / 1024**2:.2f} MB")
    print(f"显存节省: {save_pct:.2f}%")

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
    measure_memory()