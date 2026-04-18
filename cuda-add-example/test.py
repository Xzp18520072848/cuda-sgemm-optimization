import torch
import time

# 导入我们编译好的 CUDA 模块
import add_cuda

def test():
    n = 10000000  # 1000万元素
    a = torch.randn(n, device='cuda', dtype=torch.float32)
    b = torch.randn(n, device='cuda', dtype=torch.float32)

    # 预热
    _ = add_cuda.add(a, b)
    torch.cuda.synchronize()

    # 测试自定义 CUDA 算子
    start = time.time()
    c_cuda = add_cuda.add(a, b)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    # 测试 PyTorch 原生加法
    start = time.time()
    c_torch = a + b
    torch.cuda.synchronize()
    torch_time = time.time() - start

    print(f"CUDA 算子耗时: {cuda_time:.6f} 秒")
    print(f"PyTorch 耗时: {torch_time:.6f} 秒")
    print(f"结果一致: {torch.allclose(c_cuda, c_torch)}")

if __name__ == "__main__":
    test()