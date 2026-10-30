def benchmark_mps(size=10000, warmpup=True):
    import time

    torch.manual_seed(1234)
    TENSOR_A_CPU = torch.rand(size, size)
    TENSOR_B_CPU = torch.rand(size, size)

    torch.manual_seed(1234)
    TENSOR_A_MPS = torch.rand(size, size).to('mps')
    TENSOR_B_MPS = torch.rand(size, size).to('mps')

    # Warm-up (GPU perform better with this)
    if warmpup:
        for _ in range(100):
            torch.matmul(torch.rand(500,500).to('mps'), torch.rand(500,500).to('mps'))
        
    start_time = time.time()
    torch.matmul(TENSOR_A_CPU, TENSOR_B_CPU)
    print("CPU : --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    torch.matmul(TENSOR_A_MPS, TENSOR_B_MPS)
    print("MPS : --- %s seconds ---" % (time.time() - start_time))

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS device found!")
    print("MPS device: ", mps_device)
    print("Benchmarking...")
    benchmark_mps()
else:
    print ("MPS device not found.")