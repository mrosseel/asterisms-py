import torch
import time

def benchmark_matmul(device, size=2000, iterations=100):
    """Benchmark matrix multiplication on specified device"""
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(x, y)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    gflops = (2 * size**3 * iterations) / elapsed / 1e9
    return elapsed, gflops

print("=" * 60)
print(f"PyTorch GPU Benchmark - AMD Strix Halo gfx1151")
print("=" * 60)

# CPU Benchmark
print("\nCPU Benchmark (2000x2000 matrix multiply, 100 iterations):")
cpu_time, cpu_gflops = benchmark_matmul('cpu')
print(f"  Time: {cpu_time:.2f}s")
print(f"  Performance: {cpu_gflops:.2f} GFLOPS")

# GPU Benchmark
if torch.cuda.is_available():
    print("\nGPU Benchmark (2000x2000 matrix multiply, 100 iterations):")
    gpu_time, gpu_gflops = benchmark_matmul('cuda')
    print(f"  Time: {gpu_time:.2f}s")
    print(f"  Performance: {gpu_gflops:.2f} GFLOPS")

    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x")

    # Memory info
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("\n✗ GPU not available for benchmarking")

print("\n" + "=" * 60)
