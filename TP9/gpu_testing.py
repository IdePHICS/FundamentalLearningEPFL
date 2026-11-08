# In order to run this script is advisable to have your power supply connected to your laptop
# CPU should use all the parallelism available!


def benchmark_gpu(device, size=10000, warmpup=True):
    print(f"Benchmarking with size {size}x{size}")
    import time

    torch.manual_seed(1234)
    TENSOR_A_CPU = torch.rand(size, size)
    TENSOR_B_CPU = torch.rand(size, size)

    torch.manual_seed(1234)
    TENSOR_A_GPU = torch.rand(size, size).to(device)
    TENSOR_B_GPU = torch.rand(size, size).to(device)

    # Warm-up (GPU perform better with this)
    if warmpup:
        for _ in range(100):
            torch.matmul(torch.rand(500,500).to(device), torch.rand(500,500).to(device))
        
    start_time = time.time()
    torch.matmul(TENSOR_A_CPU, TENSOR_B_CPU)
    print("\tCPU : --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    torch.matmul(TENSOR_A_GPU, TENSOR_B_GPU)
    print("\tGPU : --- %s seconds ---" % (time.time() - start_time))

import torch

# Get the best device in order: cuda, mps, cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

print(f"Using device: {device}")
if device == 'cpu':
    print("Bad luck, your PyTorch version does not support GPU acceleration")
else:
    benchmark_gpu(device, 1000, warmpup=True)
    benchmark_gpu(device, 10000, warmpup=True)
    benchmark_gpu(device, 30000, warmpup=True)
