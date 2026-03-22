"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

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
    "anyfft_hipfft": False,
    "pyfftw": False,
    "cupy": False
}

try:
    import anyFFT
    DEPS["anyfft_fftw"] = anyFFT.has_backend("fftw")
    DEPS["anyfft_cufft"] = anyFFT.has_backend("cufft")
    DEPS["anyfft_hipfft"] = anyFFT.has_backend("hipfft")
    DEPS["gpufft_backend"] = anyFFT.get_gpu_backend_name()
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
        raise MemoryError(
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
        import cupy as cp
        cp.cuda.Device().synchronize()

    return time.time() - start


def run_cpu_benchmark(sizes, dtype_np, dtype_str, iterations=10):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}CPU Benchmark | Threads: 1 | {dtype_str}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    results = {"NumPy": []}
    if DEPS["pyfftw"]: results["pyFFTW"] = []
    if DEPS["anyfft_fftw"]: results["anyFFT"] = []

    successful_sizes = []

    for N in sizes:
        shape = (N, N, N)

        try:
            ensure_physical_memory(N, dtype_np)
        except MemoryError as e:
            print(f"\n  {YELLOW}[!] {e} Stopping CPU benchmark for this precision.{RESET}")
            break

        print(f"\n{CYAN}--- Benchmarking shape {shape} ---{RESET}")
        successful_sizes.append(N)

        print(f"  {'-> NumPy...':<30}", end="", flush=True)
        data_np = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(dtype_np)

        def run_numpy():
            np.fft.ifftn(np.fft.fftn(data_np))

        elapsed = measure_time(run_numpy, iterations)
        results["NumPy"].append(elapsed / iterations)
        print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        if DEPS["pyfftw"]:
            import pyfftw
            print(f"  {'-> pyFFTW...':<30}", end="", flush=True)
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
            from anyFFT import FFT, FFTW_MEASURE
            print(f"  {'-> anyFFT (FFTW)...':<30}", end="", flush=True)
            if DEPS["pyfftw"]:
                import pyfftw
                data_align = pyfftw.empty_aligned(shape, dtype=dtype_np)
            else:
                data_align = np.empty(shape, dtype=dtype_np)

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

    gpufft_label = "anyFFT (gpuFFT)"
    results = {"CuPy": []}

    if DEPS["anyfft_cufft"]:
        gpufft_label = "anyFFT (cuFFT)"
    elif DEPS["anyfft_hipfft"]:
        gpufft_label = "anyFFT (hipFFT)"

    results[gpufft_label] = []

    successful_sizes = []

    import cupy as cp
    from anyFFT import FFT

    for N in sizes:
        shape = (N, N, N)

        try:
            ensure_vram(N, dtype_np)
        except MemoryError as e:
            print(f"\n  {YELLOW}[!] {e} Stopping GPU benchmark for this precision.{RESET}")
            cp.get_default_memory_pool().free_all_blocks()
            break

        print(f"\n{CYAN}--- Benchmarking shape {shape} ---{RESET}")
        successful_sizes.append(N)

        print(f"  {'-> CuPy...':<30}", end="", flush=True)
        data_dev = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(dtype_np)

        def run_cupy():
            cp.fft.ifftn(cp.fft.fftn(data_dev))

        run_cupy() # Warmup

        elapsed = measure_time(run_cupy, iterations, is_gpu=True)
        results["CuPy"].append(elapsed / iterations)
        print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        if DEPS["anyfft_cufft"] or DEPS["anyfft_hipfft"]:
            print_str = f"-> {gpufft_label}..."
            print(f"  {print_str:<30}", end="", flush=True)
            data_dev = (cp.random.rand(*shape) + 1j * cp.random.rand(*shape)).astype(dtype_np)

            plan_fwd_gpu = FFT(shape=shape, input=data_dev, output=data_dev, dtype=dtype_str, backend="gpufft")
            plan_bwd_gpu = FFT(shape=shape, input=data_dev, output=data_dev, dtype=dtype_str, backend="gpufft")

            def run_anyfft_gpufft():
                plan_fwd_gpu.forward(data_dev, data_dev)
                plan_bwd_gpu.backward(data_dev, data_dev)

            run_anyfft_gpufft() # Warmup

            elapsed = measure_time(run_anyfft_gpufft, iterations, is_gpu=True)
            results[gpufft_label].append(elapsed / iterations)
            print(f" {iterations} iters in {GREEN}{elapsed:.8f} s{RESET}")

        data_dev = None
        plan_fwd_gpu = None
        plan_bwd_gpu = None
        cp.get_default_memory_pool().free_all_blocks()

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

    # Plot 1: Absolute Times
    ax1.set_title(f"{title_prefix} Execution Time")
    ax1.set_ylabel("Time per Round-Trip (s)")
    ax1.set_yscale("log")

    for i, (label, times) in enumerate(times_dict.items()):
        ax1.plot(sizes, times, styles[i % len(styles)], label=label, color=colors[i % len(colors)], linewidth=2 if "anyFFT" in label else 1.5)

    ax1.legend()
    format_x_axis(ax1, sizes)

    # Plot 2: Speedup Factors
    ax2.set_title(f"{title_prefix} Speedup Factors")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.set_yscale("log")
    ax2.axhline(y=1, color='black', linestyle=':', alpha=0.6, label='Baseline (1x)')

    # Identify competitors (NumPy/pyFFTW/CuPy) and targets (anyFFT variants)
    competitors = [lbl for lbl in labels if "anyFFT" not in lbl]
    targets = [lbl for lbl in labels if "anyFFT" in lbl]

    print(f"\n{CYAN}Speedups for {title_prefix}:{RESET}")
    plot_idx = 0
    for target in targets:
        t_target = np.array(times_dict[target])
        for comp in competitors:
            t_comp = np.array(times_dict[comp])

            speedup = t_comp / t_target

            line_label = f"{target} vs {comp}"
            ax2.plot(sizes, speedup, styles[plot_idx % len(styles)], label=line_label, color=colors[plot_idx % len(colors)])
            print(f"  Average {line_label} speedup: {GREEN}{np.mean(speedup):.2f}x{RESET}")
            plot_idx += 1

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

    if not DEPS["anyfft_fftw"] and not (DEPS["cupy"] and (DEPS["anyfft_cufft"] or DEPS["anyfft_hipfft"])):
        print(f"{RED}CRITICAL: Minimum requirements for benchmarking (anyFFT backends) not met. Exiting.{RESET}")
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

    print(f"{CYAN}{'='*60}\nanyFFT Benchmarking\n{'='*60}{RESET}")

    print(f"\n{CYAN}Outputs will be saved to: {plots_dir.resolve()}{RESET}")
    print(f"{CYAN}Testing Grid Sizes: {sizes}{RESET}")
    print(f"{CYAN}Testing Precision: {args.precision}{RESET}")

    for dtype_np, dtype_str, label in precisions:

        if DEPS["anyfft_fftw"]:
            s_sizes, plot_data = run_cpu_benchmark(sizes, dtype_np, dtype_str)
            if s_sizes:
                output_file = plots_dir / f"cpu_results_{dtype_str}.png"
                generate_plot(s_sizes, plot_data, f"CPU {label}", output_file)

        if DEPS["cupy"] and (DEPS["anyfft_cufft"] or DEPS["anyfft_hipfft"]):
            s_sizes, plot_data = run_gpu_benchmark(sizes, dtype_np, dtype_str)
            if s_sizes:
                output_file = plots_dir / f"gpu_results_{dtype_str}.png"
                generate_plot(s_sizes, plot_data, f"GPU {label}", output_file)

if __name__ == "__main__":
    main()
