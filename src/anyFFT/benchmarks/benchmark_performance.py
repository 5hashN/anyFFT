import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

try:
    import psutil
except ImportError:
    print(f"{RED}Error: 'psutil' is required for memory safety checks. Install it with 'pip install psutil'.{RESET}")
    sys.exit(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
thread_vars = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
]
for var in thread_vars:
    os.environ[var] = "1"

DEPS = {
    "anyfft_fftw": False,
    "anyfft_cufft": False,
    "pyfftw": False,
    "cupy": False
}

try:
    import anyFFT
    DEPS["anyfft_fftw"] = anyFFT.has_backend("fftw")
    DEPS["anyfft_cufft"] = anyFFT.has_backend("cufft")
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    DEPS["pyfftw"] = True
except ImportError:
    pass

try:
    import cupy
    DEPS["cupy"] = True
except ImportError:
    pass


def ensure_physical_memory(N, dtype_np, num_arrays=3, safety_margin=0.25):
    element_size = np.dtype(dtype_np).itemsize
    total_bytes_needed = (N ** 3) * element_size * num_arrays

    available_ram = psutil.virtual_memory().available

    if total_bytes_needed > (available_ram * safety_margin):
        needed_gb = total_bytes_needed / (1024**3)
        avail_gb = available_ram / (1024**3)
        raise MemoryError(
            f"Requires ~{needed_gb:.2f} GB RAM, but only {avail_gb:.2f} GB is safely available."
        )

def ensure_vram(N, dtype_np, num_arrays=3, safety_margin=0.50):
    import cupy as cp
    element_size = np.dtype(dtype_np).itemsize
    total_bytes_needed = (N ** 3) * element_size * num_arrays

    free_mem, _ = cp.cuda.runtime.memGetInfo()
    if total_bytes_needed > (free_mem * safety_margin):
        needed_gb = total_bytes_needed / (1024**3)
        free_gb = free_mem / (1024**3)
        raise cp.cuda.memory.OutOfMemoryError(
            f"Requires ~{needed_gb:.2f} GB VRAM, but only {free_gb:.2f} GB is free."
        )

def measure_time(target_func, iterations, is_gpu=False):
    if is_gpu:
        import cupy as cp
        cp.cuda.Device().synchronize()

    start = time.time()
    for _ in range(iterations):
        target_func()

    if is_gpu:
        cp.cuda.Device().synchronize()

    return time.time() - start


def run_cpu_benchmark(sizes, dtype_np, dtype_str, iterations=10):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}CPU Benchmark | Threads: 1 | {dtype_str}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    results = {"NumPy": [], "pyFFTW": [], "anyFFT": []}
    successful_sizes = []

    import pyfftw
    from anyFFT import FFT, FFTW_MEASURE

    for N in sizes:
        shape = (N, N, N)

        try:
            ensure_physical_memory(N, dtype_np)
        except MemoryError as e:
            print(f"\n  {YELLOW}[!] {e} Stopping CPU benchmark for this precision.{RESET}")
            break

        print(f"\n{CYAN}--- Benchmarking shape {shape} ---{RESET}")
        successful_sizes.append(N)

        print(f"  {'-> NumPy...':<15}", end="", flush=True)
        data_np = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype_np)

        def run_numpy():
            np.fft.ifftn(np.fft.fftn(data_np))

        elapsed = measure_time(run_numpy, iterations)
        results["NumPy"].append(elapsed / iterations)
        print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        if DEPS["pyfftw"]:
            print(f"  {'-> pyFFTW...':<15}", end="", flush=True)
            data_align = pyfftw.empty_aligned(shape, dtype=dtype_np)
            data_align[:] = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype_np)

            fft_fwd = pyfftw.builders.fftn(data_align, planner_effort="FFTW_MEASURE", threads=1, overwrite_input=True)
            fft_inv = pyfftw.builders.ifftn(data_align, planner_effort="FFTW_MEASURE", threads=1, overwrite_input=True)

            def run_pyfftw():
                fft_fwd()
                fft_inv()

            elapsed = measure_time(run_pyfftw, iterations)
            results["pyFFTW"].append(elapsed / iterations)
            print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        if DEPS["anyfft_fftw"]:
            print(f"  {'-> anyFFT...':<15}", end="", flush=True)
            data_align = pyfftw.empty_aligned(shape, dtype=dtype_np) if DEPS["pyfftw"] else np.empty(shape, dtype=dtype_np)
            data_align[:] = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype_np)

            plan_fwd = FFT(shape=shape, input=data_align, output=data_align, dtype=dtype_str, backend="fftw", threads=1, flags=FFTW_MEASURE)
            plan_bwd = FFT(shape=shape, input=data_align, output=data_align, dtype=dtype_str, backend="fftw", threads=1, flags=FFTW_MEASURE)

            def run_anyfft_cpu():
                plan_fwd.forward(data_align, data_align)
                plan_bwd.backward(data_align, data_align)

            elapsed = measure_time(run_anyfft_cpu, iterations)
            results["anyFFT"].append(elapsed / iterations)
            print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

    return successful_sizes, results

def run_gpu_benchmark(sizes, dtype_np, dtype_str, iterations=25):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}GPU Benchmark | {dtype_str}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    results = {"CuPy": [], "anyFFT": []}
    successful_sizes = []

    import cupy as cp
    from anyFFT import FFT

    for N in sizes:
        shape = (N, N, N)

        try:
            ensure_vram(N, dtype_np)
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"\n  {YELLOW}[!] {e} Stopping GPU benchmark for this precision.{RESET}")
            cp.get_default_memory_pool().free_all_blocks()
            break

        print(f"\n{CYAN}--- Benchmarking shape {shape} ---{RESET}")
        successful_sizes.append(N)

        print(f"  {'-> CuPy...':<15}", end="", flush=True)
        data_dev = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(dtype_np)

        def run_cupy():
            cp.fft.ifftn(cp.fft.fftn(data_dev))

        run_cupy() # Warmup

        elapsed = measure_time(run_cupy, iterations, is_gpu=True)
        results["CuPy"].append(elapsed / iterations)
        print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        if DEPS["anyfft_cufft"]:
            print(f"  {'-> anyFFT...':<15}", end="", flush=True)
            data_dev = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(dtype_np)

            plan_fwd = FFT(shape=shape, input=data_dev, output=data_dev, dtype=dtype_str, backend="cufft")
            plan_bwd = FFT(shape=shape, input=data_dev, output=data_dev, dtype=dtype_str, backend="cufft")

            def run_anyfft_gpu():
                plan_fwd.forward(data_dev, data_dev)
                plan_bwd.backward(data_dev, data_dev)

            run_anyfft_gpu() # Warmup

            elapsed = measure_time(run_anyfft_gpu, iterations, is_gpu=True)
            results["anyFFT"].append(elapsed / iterations)
            print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

    return successful_sizes, results

def format_x_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes])
    ax.set_xlabel("Cube Size (N^3)")
    ax.grid(True)

def generate_plot(sizes, times_dict, title_prefix, file_path):
    times_dict = {k: v for k, v in times_dict.items() if v}
    if not times_dict:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = list(times_dict.keys())
    colors = ["red", "orange", "blue", "green", "purple"]
    styles = ["o--", "s-", "^-", "*-", "x-"]

    anyfft_idx = labels.index("anyFFT") if "anyFFT" in labels else -1

    # Plot 1: Absolute Times
    ax1.set_title(f"{title_prefix} Execution Time")
    ax1.set_ylabel("Time per Round-Trip (s)")
    ax1.set_yscale("log")

    for i, (label, times) in enumerate(times_dict.items()):
        ax1.plot(sizes, times, styles[i], label=label, color=colors[i], linewidth=2 if label == "anyFFT" else 1.5)

    ax1.legend()
    format_x_axis(ax1, sizes)

    # Plot 2: Speedup Factors
    ax2.set_title(f"{title_prefix} Speedup Factors")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.set_yscale("log")
    ax2.axhline(y=1, color='black', linestyle=':', alpha=0.6, label='Baseline (1x)')

    if anyfft_idx != -1 and times_dict.get("anyFFT"):
        t_any = np.array(times_dict["anyFFT"])
        print()
        for i, (label, times) in enumerate(times_dict.items()):
            if label != "anyFFT":
                speedup = np.array(times) / t_any
                ax2.plot(sizes, speedup, styles[i], label=f"anyFFT vs {label}", color=colors[i])
                print(f"  Average anyFFT speedup vs {label}: {GREEN}{np.mean(speedup):.2f}x{RESET}")

        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()

    format_x_axis(ax2, sizes)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)
    print(f"\n{GREEN}Graph saved to '{file_path}'{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="anyFFT Performance Benchmark")
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="Maximum grid size to benchmark. Tests powers of 2 up to this value."
    )
    parser.add_argument(
        "--precision",
        choices=["single", "double", "both"],
        default="both",
        help="Select the precision to benchmark: 'single' (complex64), 'double' (complex128), or 'both'."
    )
    args = parser.parse_args()

    if not (DEPS["pyfftw"] and DEPS["anyfft_fftw"]) and not (DEPS["cupy"] and DEPS["anyfft_cufft"]):
        print(f"{RED}CRITICAL: Minimum requirements for CPU or GPU benchmarking not met. Exiting.{RESET}")
        sys.exit(1)

    sizes = []
    current_size = 32
    while current_size <= args.max_size:
        sizes.append(current_size)
        current_size *= 2

    if not sizes:
        print(f"{RED}Error: --max-size ({args.max_size}) is too small. Minimum size is 32.{RESET}")
        sys.exit(1)

    all_precisions = [
        (np.complex64, "complex64", "Single Precision"),
        (np.complex128, "complex128", "Double Precision")
    ]

    if args.precision == "single":
        precisions = [all_precisions[0]]
    elif args.precision == "double":
        precisions = [all_precisions[1]]
    else:
        precisions = all_precisions

    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{CYAN}Outputs will be saved to: {plots_dir.resolve()}{RESET}")
    print(f"{CYAN}Testing Grid Sizes: {sizes}{RESET}")
    print(f"{CYAN}Testing Precision: {args.precision}{RESET}")

    for dtype_np, dtype_str, label in precisions:

        if DEPS["pyfftw"] and DEPS["anyfft_fftw"]:
            s_sizes, plot_data = run_cpu_benchmark(sizes, dtype_np, dtype_str)
            if s_sizes:
                output_file = plots_dir / f"cpu_results_{dtype_str}.png"
                generate_plot(s_sizes, plot_data, f"CPU {label}", output_file)

        if DEPS["cupy"] and DEPS["anyfft_cufft"]:
            s_sizes, plot_data = run_gpu_benchmark(sizes, dtype_np, dtype_str)
            if s_sizes:
                output_file = plots_dir / f"gpu_results_{dtype_str}.png"
                generate_plot(s_sizes, plot_data, f"GPU {label}", output_file)

if __name__ == "__main__":
    main()
