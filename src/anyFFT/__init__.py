"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

_HAS_FFTW = False
_HAS_FFTW_MPI = False
_HAS_CUDA = False
_HAS_CUDA_MPI = False

try:
    from ._core import fftw_serial
    _HAS_FFTW = True
except ImportError:
    pass

try:
    from ._core import fftw_mpi
    _HAS_FFTW_MPI = True
except ImportError:
    pass

try:
    from ._core import FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE
except ImportError:
    FFTW_ESTIMATE = 64
    FFTW_MEASURE = 0
    FFTW_PATIENT = 32
    FFTW_EXHAUSTIVE = 8

try:
    from ._core import cufft_serial
    _HAS_CUDA = True
except ImportError:
    pass

try:
    from ._core import cufft_mpi
    _HAS_CUDA_MPI = True
except ImportError:
    pass

# Define Exports
__all__ = ["FFT"]
if _HAS_FFTW:
    __all__.extend(["fftw_serial"])
if _HAS_FFTW_MPI:
    __all__.extend(["fftw_mpi"])
if _HAS_CUDA:
    __all__.extend(["cufft_serial"])
if _HAS_CUDA_MPI:
    __all__.extend(["cufft_mpi"])


def has_backend(backend_name):
    """
    Checks if a specific backend is compiled and available.

    Args:
        backend_name (str): 'fftw', 'fftw_mpi', 'cufft', 'cufft_mpi'.

    Returns:
        bool: True if available, False otherwise.
    """
    if backend_name == "fftw":
        return _HAS_FFTW
    if backend_name == "fftw_mpi":
        return _HAS_FFTW_MPI
    if backend_name == "cufft":
        return _HAS_CUDA
    if backend_name == "cufft_mpi":
        return _HAS_CUDA_MPI
    return False


def FFT(
    ndim=None,
    shape=None,
    axes=None,
    input=None,
    output=None,
    dtype="float64",
    backend="fftw",
    comm=None,
    grid=None,
    threads=1,
    flags=None,
):
    """
    Factory function to create an FFT backend instance.

    Args:
        ndim (int): Number of dimensions (e.g., 2 or 3).
        shape (list/tuple): Dimensions of the transform. Must be global shape for MPI backends.
        axes (list/tuple, optional): Specific axes to transform.
        input (array, optional): Input array (Required for FFTW planning).
        output (array, optional): Output array (Required for FFTW planning).
        dtype (str): 'float32', 'float64', 'complex64', or 'complex128'.
        backend (str): 'fftw', 'fftw_mpi', 'cufft', or 'cufft_mpi'.
        comm (mpi4py.MPI.Comm, optional): MPI Communicator (Required for MPI backends).
        grid (list/tuple, optional): Process grid [p0, p1] for Slab/Pencil decomposition. Defaults to None (1D Slab). (cufft_mpi backend only).
        n_threads (int, optional): Number of OpenMP threads to use (FFTW backend only). Defaults to 1.
        flags (int, optional): FFTW planner flags (e.g., anyFFT.FFTW_MEASURE). Defaults to anyFFT.FFTW_ESTIMATE. (fftw backend only).

    Returns:
        An instance of the requested FFT backend.
    """

    if comm is not None:
        # Handle mpi4py object -> Fortran Integer Handle conversion
        comm_handle = comm
        if hasattr(comm, "py2f"):
            comm_handle = comm.py2f()

        if not isinstance(comm_handle, int):
            raise TypeError("Communicator must be an mpi4py communicator or an integer handle.")

    if grid is None:
        grid = []


    if isinstance(ndim, list) or isinstance(ndim, tuple):
        shape = ndim

    if not isinstance(shape, list) and not isinstance(shape, tuple):
        raise ValueError("Must provide valid 'shape' to create a plan.")

    ndim = len(shape)

    if axes is None:
        axes = []

    if not isinstance(dtype, str):
        dtype = str(dtype).split("'")[1]
        dtype = str(dtype).split(".")[-1]

    if dtype not in ["float32", "float64", "complex64", "complex128"]:
        raise ValueError("Invalid 'dtype' specified. Must be one of: 'float32', 'float64', 'complex64', 'complex128'.")


    if backend == "fftw":
        if flags is None:
            flags = FFTW_ESTIMATE

        if comm is None:
            if not _HAS_FFTW:
                raise RuntimeError("The 'fftw' backend was requested, but anyFFT was compiled without FFTW support.")

            if input is None or output is None:
                raise ValueError("The 'fftw' backend requires 'input' and 'output' dummy arrays to create a plan.")

            return fftw_serial(shape, axes, input, output, dtype, threads, flags)

        else:
            if not _HAS_FFTW_MPI:
                raise RuntimeError("The 'fftw_mpi' backend was requested, but anyFFT was compiled without FFTW-MPI support.")

            if input is None or output is None:
                raise ValueError("The 'fftw_mpi' backend requires 'input' and 'output' (local) arrays to create a plan.")

            return fftw_mpi(shape, input, output, dtype, comm_handle)

    elif backend == "cufft":
        if comm is None:
            if not _HAS_CUDA:
                raise RuntimeError("The 'cufft' backend was requested, but anyFFT was compiled without CUDA support.")

            return cufft_serial(shape, axes, dtype)

        else:
            if not _HAS_CUDA_MPI:
                raise RuntimeError("The 'cufft_mpi' backend was requested, but anyFFT was compiled without CUDA support.")

            if input is None or output is None:
                raise ValueError("The 'cufft_mpi' backend requires 'input' and 'output' arrays to check for In-Place/Out-of-Place consistency.")

            return cufft_mpi(shape, grid, input, output, dtype, comm_handle)

    else:
        raise ValueError(f"Unknown backend type: '{backend}'. Available: {__all__[1:]}")
