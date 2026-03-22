"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import numpy as np
import sys

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_header(backend_name, comm=None):
    rank = 0
    size = 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()

    if rank == 0:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}Testing anyFFT Backend: {backend_name}{RESET}")
        if comm is not None:
            print(f"{CYAN}MPI Size: {size}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        sys.stdout.flush()

def print_section(title, rank=0):
    if rank == 0:
        print(f"\n{CYAN}--- {title} ---{RESET}")
        sys.stdout.flush()

def print_test_start(test_name, dtype, rank=0):
    """
    Prints the start of a test line (e.g., '  [Gen C2C] ...').
    """
    if rank == 0:
        print(f"  {test_name:<45} {dtype:<10} ...", end=" ")
        sys.stdout.flush()

def print_test_result(diff, tol=1e-4, rank=0, error=None):
    """
    Prints colored PASSED/FAILED based on the diff.
    Returns True if passed, False otherwise, for global tracking.
    """
    passed = False
    if error:
        msg = f"{RED}FAILED{RESET} (Error: {error})"
    elif diff < tol:
        msg = f"{GREEN}PASSED{RESET} (Diff: {diff:.2e})"
        passed = True
    else:
        msg = f"{RED}FAILED{RESET} (Diff: {diff:.2e})"

    if rank == 0:
        print(msg)
        sys.stdout.flush()

    return passed

def print_skipped(reason, rank=0):
    """Prints a SKIPPED status."""
    if rank == 0:
        print(f"{YELLOW}SKIPPED{RESET} ({reason})")
        sys.stdout.flush()

def print_summary(total, passed, failed_list, rank=0):
    """Unified summary block for both Serial and MPI tests."""
    if rank == 0:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"SUMMARY: Total {total} | {GREEN}Passed {passed}{RESET} | {RED}Failed {len(failed_list)}{RESET}")
        if failed_list:
            print(f"\n{RED}Failures:{RESET}")
            for f in failed_list:
                print(f"  - {f}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        sys.stdout.flush()

def get_c2c_dtype_str(real_dtype_str):
    """
    Maps a real precision string (e.g., 'float64') to the corresponding
    complex string required by the C++ library constructor ('complex128').
    """
    if real_dtype_str == "float64":
        return "complex128"
    elif real_dtype_str == "float32":
        return "complex64"
    elif real_dtype_str == "complex128":
        return "complex128"
    elif real_dtype_str == "complex64":
        return "complex64"
    else:
        raise ValueError(f"Unknown precision: {real_dtype_str}")


def get_numpy_types(dtype_str):
    """
    Returns a tuple (real_dtype, complex_dtype) for NumPy.
    Accepts 'floatXX' or 'complexXX' strings.
    """
    if dtype_str in ["float32", "complex64"]:
        return np.float32, np.complex64
    elif dtype_str in ["float64", "complex128"]:
        return np.float64, np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_cupy_types(dtype_str):
    """
    Returns a tuple (real_dtype, complex_dtype) for CuPy.
    Used by GPU backends.
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is not installed, cannot determine GPU types.")

    if dtype_str in ["float32", "complex64"]:
        return cp.float32, cp.complex64
    elif dtype_str in ["float64", "complex128"]:
        return cp.float64, cp.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def generate_data(shape, dtype, start_indices=None, use_gpu=False):
    """
    Generates deterministic test data (Sin/Cos waves).
    Can generate directly on GPU if use_gpu=True and CuPy is available.

    Arguments:
        shape: The local shape to generate data for.
        dtype: The numpy/cupy dtype of the output array.
        start_indices: (Optional) For MPI, the global offsets [dim0_start, dim1_start, ...].
                       Defaults to [0, 0, ...].
        use_gpu: (bool) If True, generate data on GPU using CuPy.

    Returns:
        A NumPy or CuPy array with deterministic sin/cos wave data.
    """
    if use_gpu:
        if not HAS_CUPY:
            raise ImportError("GPU generation requested but CuPy is not installed.")
        xp = cp
    else:
        xp = np

    if start_indices is None:
        start_indices = [0] * len(shape)

    # Create global coordinate grids for the local slice
    ranges = []
    for i in range(len(shape)):
        start = start_indices[i]
        length = shape[i]
        ranges.append(xp.arange(start, start + length))

    # Use the selected backend (np or cp) for meshgrid
    grids = xp.meshgrid(*ranges, indexing="ij")

    data = xp.zeros(shape, dtype=dtype)

    # Check if complex using dtype kind to be safe across backends
    is_complex = np.dtype(dtype).kind == "c"

    # Create a pattern: sin(x) + cos(y) + sin(z)...
    # Complex: (sin(x) + i*cos(x)) + ...
    if is_complex:
        # Base wave on 1st dimension
        data += xp.sin(grids[0]) + 1j * xp.cos(grids[0])

        # Add waves from other dimensions to make it fully 3D/ND
        if len(grids) > 1:
            data += xp.cos(grids[1]) + 1j * xp.sin(grids[1])
        if len(grids) > 2:
            data += xp.sin(grids[2]) + 1j * xp.cos(grids[2])
        if len(grids) > 3:
            data += xp.cos(grids[3]) + 1j * xp.sin(grids[3])

    else:  # Real
        data += xp.sin(grids[0])
        if len(grids) > 1:
            data += xp.cos(grids[1])
        if len(grids) > 2:
            data += xp.sin(grids[2])
        if len(grids) > 3:
            data += xp.cos(grids[3])

    return data
