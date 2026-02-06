import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing

# Try imports
try:
    import anyFFT
    from anyFFT import FFT
except ImportError:
    print("CRITICAL: anyFFT not found. Install it first via 'pip install .'")
    sys.exit(1)

try:
    import pyfftw
    HAS_PYFFTW = True
    pyfftw.interfaces.cache.enable() # Allow pyfftw to cache plans like scipy
except ImportError:
    HAS_PYFFTW = False
    print("Notice: pyfftw not found (skipping CPU competitor)")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Notice: cupy not found (skipping GPU benchmarks)")

def benchmark_cpu_3d(sizes, iterations=10):
    n_threads = multiprocessing.cpu_count()
    print("\n" + "="*60)
    print(f"Starting CPU Benchmark (3D Complex) | Threads: {n_threads}")
    print("="*60)

    times_numpy = []
    times_pyfftw = []
    times_anyfft = []

    for N in sizes:
        shape = (N, N, N)
        print(f"Benchmarking shape {shape}...", end="", flush=True)

        # Data Prep
        data_in = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        data_in = data_in.astype(np.complex128)
        data_out = np.zeros_like(data_in)

        # NumPy uses MKL/BLAS threads implicitly, so we don't set arguments here.
        start = time.time()
        for _ in range(iterations):
            _ = np.fft.fftn(data_in)
        times_numpy.append((time.time() - start) / iterations)

        # pyFFTW
        if HAS_PYFFTW:
            # We explicitly enable threads for pyFFTW to match anyFFT
            fft_obj = pyfftw.builders.fftn(data_in,
                                           planner_effort='FFTW_ESTIMATE',
                                           threads=n_threads)
            start = time.time()
            for _ in range(iterations):
                fft_obj() # Execute
            times_pyfftw.append((time.time() - start) / iterations)

        # anyFFT
        # We pass the thread count to the constructor
        fft = FFT(ndim=3, shape=shape, input=data_in, output=data_out, dtype="complex128",
                  backend="fftw", threads=n_threads, flags=anyFFT.FFTW_MEASURE)

        start = time.time()
        for _ in range(iterations):
            fft.forward(data_in, data_out)
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_numpy, times_pyfftw, times_anyfft

def benchmark_gpu_3d(sizes, iterations=20):
    if not HAS_CUPY:
        return [], []

    print("\n" + "="*60)
    print("Starting GPU Benchmark (3D Complex)")
    print("="*60)

    times_cupy = []
    times_anyfft = []

    for N in sizes:
        shape = (N, N, N)
        print(f"Benchmarking shape {shape}...", end="", flush=True)

        # Data Prep
        data_host = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        data_in = cp.asarray(data_host.astype(np.complex128))
        data_out = cp.zeros_like(data_in)

        # Warmup
        cp.fft.fftn(data_in)
        cp.cuda.Device().synchronize()

        # CuPy
        start = time.time()
        for _ in range(iterations):
            _ = cp.fft.fftn(data_in)
        cp.cuda.Device().synchronize() # Wait for GPU
        times_cupy.append((time.time() - start) / iterations)

        # anyFFT
        # Threads arg is ignored for GPU backend, but safe to pass if generic
        fft = FFT(ndim=3, shape=shape, input=data_in, output=data_out,
                  dtype="complex128", backend="cufft")

        # Warmup anyFFT
        fft.forward(data_in, data_out)
        cp.cuda.Device().synchronize()

        start = time.time()
        for _ in range(iterations):
            fft.forward(data_in, data_out)
        cp.cuda.Device().synchronize() # Wait for GPU
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_cupy, times_anyfft

def plot_results(sizes, cpu_res, gpu_res):
    t_np, t_pyfftw, t_any_cpu = cpu_res
    t_cp, t_any_gpu = gpu_res

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: CPU Battle
    ax1.set_title("CPU FFT Performance (Lower is Better)")
    ax1.plot(sizes, t_np, 'o--', label='NumPy', color='red')
    if t_pyfftw:
        ax1.plot(sizes, t_pyfftw, 's-', label='pyFFTW (Builder)', color='orange')
    ax1.plot(sizes, t_any_cpu, '^-', label='anyFFT (FFTW)', color='blue', linewidth=2)
    ax1.set_xlabel("Cube Size (N^3)")
    ax1.set_ylabel("Time per call (s)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Speedup Factors
    ax2.set_title("Speedup vs NumPy (Higher is Better)")

    # Avoid div by zero
    np_arr = np.array(t_np)

    if t_any_cpu:
        speedup_cpu = np_arr / np.array(t_any_cpu)
        ax2.plot(sizes, speedup_cpu, '^-', label='anyFFT (CPU) Speedup', color='blue')

    if t_any_gpu and t_cp:
        # We compare GPU time against CPU NumPy to show "Why use GPU"
        speedup_gpu_vs_cpu = np_arr / np.array(t_any_gpu)
        ax2.plot(sizes, speedup_gpu_vs_cpu, '*-', label='anyFFT (GPU) vs NumPy', color='green')

        # Optional: Print raw comparison for GPU
        speedup_gpu_vs_cupy = np.array(t_cp) / np.array(t_any_gpu)
        print(f"\nAverage anyFFT GPU speedup over CuPy: {np.mean(speedup_gpu_vs_cupy):.2f}x")

    ax2.set_xlabel("Cube Size (N^3)")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log') # Log scale is better for GPU speedups

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nResults saved to 'benchmark_results.png'")

if __name__ == "__main__":
    SIZES = [32, 64, 96, 128, 256]

    cpu_results = benchmark_cpu_3d(SIZES)
    gpu_results = benchmark_gpu_3d(SIZES)

    plot_results(SIZES, cpu_results, gpu_results)