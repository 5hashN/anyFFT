import time
import numpy as np
import matplotlib.pyplot as plt
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
    pyfftw.interfaces.cache.enable()  # Allow pyfftw to cache plans like scipy
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
    print("\n" + "=" * 60)
    print(f"Starting CPU Benchmark (3D Complex) | Threads: {n_threads}")
    print("=" * 60)

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
            fft_obj = pyfftw.builders.fftn(
                data_in, planner_effort="FFTW_MEASURE", threads=n_threads
            )
            start = time.time()
            for _ in range(iterations):
                fft_obj()  # Execute
            times_pyfftw.append((time.time() - start) / iterations)

        # anyFFT
        # We pass the thread count to the constructor
        # Note: Using FFTW_MEASURE for maximum performance in benchmark
        fft = FFT(ndim=3, shape=shape, input=data_in, output=data_out,
                  dtype="complex128", backend="fftw",
                  threads=n_threads, flags=anyFFT.FFTW_MEASURE)

        start = time.time()
        for _ in range(iterations):
            fft.forward(data_in, data_out)
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_numpy, times_pyfftw, times_anyfft


def benchmark_gpu_3d(sizes, iterations=20):
    if not HAS_CUPY:
        return [], []

    print("\n" + "=" * 60)
    print("Starting GPU Benchmark (3D Complex)")
    print("=" * 60)

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
        cp.cuda.Device().synchronize()  # Wait for GPU
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
        cp.cuda.Device().synchronize()  # Wait for GPU
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_cupy, times_anyfft


def plot_results(sizes, cpu_res, gpu_res):
    t_np, t_pyfftw, t_any_cpu = cpu_res
    t_cp, t_any_gpu = gpu_res

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Absolute Times
    ax1.set_title("FFT Execution Time (Lower is Better)")
    ax1.plot(sizes, t_np, "o--", label="NumPy", color="red", alpha=0.7)
    if t_pyfftw:
        ax1.plot(sizes, t_pyfftw, "s-", label="pyFFTW", color="orange")
    ax1.plot(sizes, t_any_cpu, "^-", label="anyFFT (FFTW)", color="blue", linewidth=2)

    if t_any_gpu:
        # Show GPU raw times on the first plot too for perspective
        ax1.plot(sizes, t_any_gpu, "*-", label="anyFFT (GPU)", color="green")

    ax1.set_xlabel("Cube Size (N^3)")
    ax1.set_ylabel("Time per call (s)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale("log") # Log scale helps view CPU vs GPU differences

    # Plot 2: Speedup Factors
    ax2.set_title("Speedup Factors (Higher is Better)")

    # Base Arrays
    t_np_arr = np.array(t_np)
    t_any_cpu_arr = np.array(t_any_cpu)

    # anyFFT (CPU) vs NumPy
    if len(t_any_cpu) > 0:
        speedup_vs_numpy = t_np_arr / t_any_cpu_arr
        ax2.plot(sizes, speedup_vs_numpy, "^-", label="anyFFT vs NumPy (CPU)", color="blue")

    # anyFFT (CPU) vs pyFFTW
    if t_pyfftw and len(t_pyfftw) == len(t_any_cpu):
        t_pyfftw_arr = np.array(t_pyfftw)
        speedup_vs_pyfftw = t_pyfftw_arr / t_any_cpu_arr
        ax2.plot(sizes, speedup_vs_pyfftw, "s--", label="anyFFT vs pyFFTW (CPU)", color="orange")

    # GPU Speedups
    if t_any_gpu and t_cp:
        t_any_gpu_arr = np.array(t_any_gpu)
        t_cp_arr = np.array(t_cp)

        # anyFFT (GPU) vs NumPy (Baseline)
        speedup_gpu_vs_cpu = t_np_arr / t_any_gpu_arr
        ax2.plot(
            sizes,
            speedup_gpu_vs_cpu,
            "*-",
            label="anyFFT (GPU) vs NumPy",
            color="green",
        )

        # anyFFT (GPU) vs CuPy
        speedup_gpu_vs_cupy = t_cp_arr / t_any_gpu_arr
        ax2.plot(
            sizes,
            speedup_gpu_vs_cupy,
            "x--",
            label="anyFFT (GPU) vs CuPy",
            color="purple"
        )

        print(
            f"\nAverage anyFFT GPU speedup over CuPy: {np.mean(speedup_gpu_vs_cupy):.2f}x"
        )

    ax2.set_xlabel("Cube Size (N^3)")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")  # Log scale is better for GPU speedups

    plt.tight_layout()
    plt.savefig("benchmarks/benchmark_results.png")
    print("\nResults saved to 'benchmark_results.png'")


if __name__ == "__main__":
    SIZES = [32, 64, 96, 128, 256]

    cpu_results = benchmark_cpu_3d(SIZES)
    gpu_results = benchmark_gpu_3d(SIZES)

    plot_results(SIZES, cpu_results, gpu_results)