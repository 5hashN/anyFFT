import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import sys

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
    pyfftw.interfaces.cache.enable()
except ImportError:
    HAS_PYFFTW = False
    print("Notice: pyfftw not found (skipping CPU competitor)")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Notice: cupy not found (skipping GPU benchmarks)")


def benchmark_cpu_3d(sizes, iterations=25):
    n_threads = multiprocessing.cpu_count()
    print("\n" + "=" * 60)
    print(f"Starting CPU Benchmark (3D Forward+Backward) | Threads: {n_threads}")
    print("=" * 60)

    times_numpy = []
    times_pyfftw = []
    times_anyfft = []

    for N in sizes:
        shape = (N, N, N)
        print(f"Benchmarking shape {shape}...", end="", flush=True)

        # NumPy (Forward + Backward)
        # Data Prep
        data = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(np.complex128)

        start = time.time()
        for _ in range(iterations):
            # forward -> backward (round trip)
            data = np.fft.ifftn(np.fft.fftn(data))
        times_numpy.append((time.time() - start) / iterations)

        # pyFFTW (Forward + Backward In-Place)
        if HAS_PYFFTW:
            data = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(np.complex128)

            # Plan Forward
            fft_fwd = pyfftw.builders.fftn(
                data, planner_effort="FFTW_MEASURE", threads=n_threads, overwrite_input=True
            )
            # Plan Backward (Inverse) - strictly in-place on the same data array
            fft_inv = pyfftw.builders.ifftn(
                data, planner_effort="FFTW_MEASURE", threads=n_threads, overwrite_input=True
            )

            start = time.time()
            for _ in range(iterations):
                fft_fwd()
                fft_inv()
            times_pyfftw.append((time.time() - start) / iterations)

        # anyFFT (Forward + Backward In-Place)
        data = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(np.complex128)

        # We need two plans: one for forward, one for backward
        # Both point to the same 'data' buffer for input and output
        plan_fwd = FFT(ndim=3, shape=shape, input=data, output=data,
                       dtype="complex128", backend="fftw",
                       threads=n_threads, flags=anyFFT.FFTW_MEASURE)

        plan_bwd = FFT(ndim=3, shape=shape, input=data, output=data,
                       dtype="complex128", backend="fftw",
                       threads=n_threads, flags=anyFFT.FFTW_MEASURE)

        start = time.time()
        for _ in range(iterations):
            plan_fwd.forward(data, data)
            plan_bwd.backward(data, data)
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_numpy, times_pyfftw, times_anyfft


def benchmark_gpu_3d(sizes, iterations=10):
    if not HAS_CUPY:
        print("Skipping GPU benchmark (CuPy not found)")
        return [], []

    print("\n" + "=" * 60)
    print("Starting GPU Benchmark (3D Forward+Backward)")
    print("=" * 60)

    times_cupy = []
    times_anyfft = []

    for N in sizes:
        shape = (N, N, N)
        print(f"Benchmarking shape {shape}...", end="", flush=True)

        # CuPy (Forward + Backward)
        data_host = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(np.complex128)
        data_dev = cp.asarray(data_host)

        # Warmup
        cp.fft.ifftn(cp.fft.fftn(data_dev))
        cp.cuda.Device().synchronize()

        start = time.time()
        for _ in range(iterations):
            data_dev = cp.fft.ifftn(cp.fft.fftn(data_dev))
        cp.cuda.Device().synchronize()
        times_cupy.append((time.time() - start) / iterations)

        # anyFFT (Forward + Backward In-Place)
        data_dev = cp.asarray(data_host) # Fresh copy

        # Plan Forward
        plan_fwd = FFT(ndim=3, shape=shape, input=data_dev, output=data_dev,
                       dtype="complex128", backend="cufft")
        # Plan Backward
        plan_bwd = FFT(ndim=3, shape=shape, input=data_dev, output=data_dev,
                       dtype="complex128", backend="cufft")

        # Warmup
        plan_fwd.forward(data_dev, data_dev)
        plan_bwd.backward(data_dev, data_dev)
        cp.cuda.Device().synchronize()

        start = time.time()
        for _ in range(iterations):
            plan_fwd.forward(data_dev, data_dev)
            plan_bwd.backward(data_dev, data_dev)

        cp.cuda.Device().synchronize()
        times_anyfft.append((time.time() - start) / iterations)

        print(" Done.")

    return times_cupy, times_anyfft


def plot_results(sizes, cpu_res=None, gpu_res=None):
    t_np, t_pyfftw, t_any_cpu = cpu_res if cpu_res else ([], [], [])
    t_cp, t_any_gpu = gpu_res if gpu_res else ([], [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Absolute Times
    ax1.set_title("Round-Trip Execution Time (Fwd + Bwd)")
    if t_np: ax1.plot(sizes, t_np, "o--", label="NumPy", color="red", alpha=0.7)
    if t_pyfftw: ax1.plot(sizes, t_pyfftw, "s-", label="pyFFTW", color="orange")
    if t_any_cpu: ax1.plot(sizes, t_any_cpu, "^-", label="anyFFT (CPU)", color="blue", linewidth=2)
    if t_any_gpu: ax1.plot(sizes, t_any_gpu, "*-", label="anyFFT (GPU)", color="green")
    if t_cp: ax1.plot(sizes, t_cp, "x-", label="CuPy", color="purple", alpha=0.5)

    ax1.set_xlabel("Cube Size (N^3)")
    ax1.set_ylabel("Time per Round-Trip (s)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale("log")

    # Plot 2: Speedup Factors
    ax2.set_title("Speedup Factors (Higher is Better)")

    t_any_cpu_arr = np.array(t_any_cpu) if t_any_cpu else np.array([])
    t_any_gpu_arr = np.array(t_any_gpu) if t_any_gpu else np.array([])

    # CPU Comparisons
    if len(t_any_cpu_arr) > 0:
        if t_np:
            speedup = np.array(t_np) / t_any_cpu_arr
            ax2.plot(sizes, speedup, "^-", label="anyFFT vs NumPy (CPU)", color="blue")
            print(f"anyFFT (CPU) vs NumPy:  {np.mean(speedup):.2f}x")

        if t_pyfftw:
            speedup = np.array(t_pyfftw) / t_any_cpu_arr
            ax2.plot(sizes, speedup, "s--", label="anyFFT vs pyFFTW (CPU)", color="orange")
            print(f"anyFFT (CPU) vs pyFFTW: {np.mean(speedup):.2f}x")

    # GPU Comparisons
    if len(t_any_gpu_arr) > 0:
        if t_np:
            ax2.plot(sizes, np.array(t_np) / t_any_gpu_arr, "*-", label="anyFFT (GPU) vs NumPy", color="green")
        if t_cp:
            speedup = np.array(t_cp) / t_any_gpu_arr
            ax2.plot(sizes, speedup, "x--", label="anyFFT (GPU) vs CuPy", color="purple")
            print(f"\nAverage anyFFT GPU speedup over CuPy: {np.mean(speedup):.2f}x")

    ax2.set_xlabel("Cube Size (N^3)")
    ax2.set_ylabel("Speedup Factor (x)")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("benchmarks/benchmark_results.png")
    print("\nResults saved to 'benchmark_results.png'")


if __name__ == "__main__":
    SIZES = [32, 64, 128, 256]

    cpu_results = None
    gpu_results = None

    try:
        cpu_results = benchmark_cpu_3d(SIZES)
    except:
        print("Skipping CPU benchmarks (RUN_CPU=False)...")

    try:
        gpu_results = benchmark_gpu_3d(SIZES)
    except:
        print("Skipping GPU benchmarks (RUN_GPU=False)...")

    plot_results(SIZES, cpu_results, gpu_results)
