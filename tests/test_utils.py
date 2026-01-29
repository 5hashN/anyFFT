import numpy as np
import sys

# Try to import CuPy for GPU backends (cufft / cufftmp)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Homogeneous Printing Helpers
def print_header(backend_name, comm=None):
    """Prints the main test header."""
    rank = 0
    size = 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()

    if rank == 0:
        print(f"========================================")
        print(f"Testing anyFFT Backend: {backend_name}")
        if comm is not None:
            print(f"MPI Size: {size}")
        print(f"========================================\n")
        sys.stdout.flush()

def print_config(ndim, shape, rank=0):
    """Prints the configuration separator."""
    if rank == 0:
        print(f"----------------------------------------")
        print(f"Config: {ndim}D | Shape: {list(shape)}")
        print(f"----------------------------------------")
        sys.stdout.flush()

def print_test_start(test_name, dtype, rank=0):
    """Prints the start of a test line."""
    if rank == 0:
        # Fixed width for test name to align columns
        print(f"  {test_name:<18} {dtype:<10} ...", end=" ")
        sys.stdout.flush()

def print_test_result(diff, tol=1e-4, rank=0, error=None):
    """Prints PASSED/FAILED based on diff."""
    if rank == 0:
        if error:
            print(f"FAILED (Error: {error})")
        elif diff < tol:
            print(f"PASSED (Diff: {diff:.2e})")
        else:
            print(f"FAILED (Diff: {diff:.2e})")
        sys.stdout.flush()

def print_skipped(reason, rank=0):
    if rank == 0:
        print(f"SKIPPED ({reason})")
        sys.stdout.flush()

# Type & Data Helpers
def get_c2c_dtype_str(real_dtype_str):
    """Maps 'float64' -> 'complex128' for the library constructor."""
    if real_dtype_str == "float64":
        return "complex128"
    elif real_dtype_str == "float32":
        return "complex64"
    else:
        raise ValueError(f"Unknown precision: {real_dtype_str}")

def get_numpy_types(dtype_str):
    """Returns (real_dtype, complex_dtype) numpy types."""
    if dtype_str == "float32" or dtype_str == "complex64":
        return np.float32, np.complex64
    elif dtype_str == "float64" or dtype_str == "complex128":
        return np.float64, np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def get_cupy_types(dtype_str):
    """
    Returns (real_dtype, complex_dtype) cupy types.
    Shared by 'cufft' (Serial) and 'cufftmp' (MPI).
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is not installed, cannot determine GPU types.")

    if dtype_str == "float32" or dtype_str == "complex64":
        return cp.float32, cp.complex64
    elif dtype_str == "float64" or dtype_str == "complex128":
        return cp.float64, cp.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def generate_data(shape, dtype, start_indices=None):
    """
    Generates deterministic data on the CPU (NumPy).

    Arguments:
        shape: The local shape to generate.
        dtype: The numpy dtype.
        start_indices: (Optional) For MPI, the global offsets [z_start, y_start, x_start].

    Returns:
        A NumPy array with deterministic sin/cos wave data.
    """
    if start_indices is None:
        start_indices = [0] * len(shape)

    # Create coordinate grids
    ranges = []
    for i in range(len(shape)):
        start = start_indices[i]
        length = shape[i]
        ranges.append(np.arange(start, start + length))

    # Generate meshgrid (sparse to save memory, or dense if needed)
    grids = np.meshgrid(*ranges, indexing='ij')

    data = np.zeros(shape, dtype=dtype)
    is_complex = np.issubdtype(dtype, np.complexfloating)

    if is_complex:
        data += (np.sin(grids[0]) + 1j*np.cos(grids[0]))
        if len(grids) > 1: data += np.cos(grids[1])
        if len(grids) > 2: data += np.sin(grids[2])
    else:
        data += np.sin(grids[0])
        if len(grids) > 1: data += np.cos(grids[1])
        if len(grids) > 2: data += np.sin(grids[2])

    return data