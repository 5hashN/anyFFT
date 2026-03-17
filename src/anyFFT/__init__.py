"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

from typing import Optional, Sequence, Union, Any

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
__all__ = ["FFT", "fft", "ifft"]

__backends__ = []
if _HAS_FFTW:
    __backends__.extend(["fftw_serial"])
if _HAS_FFTW_MPI:
    __backends__.extend(["fftw_mpi"])
if _HAS_CUDA:
    __backends__.extend(["cufft_serial"])
if _HAS_CUDA_MPI:
    __backends__.extend(["cufft_mpi"])

__all__.extend(__backends__)

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


# Context Manager
class FFTPlan:
    """
    Stateful wrapper around the C++ backend.
    Supports context manager protocol for automatic resource cleanup.
    """
    def __init__(self, backend_obj: Any) -> None:
        self._backend = backend_obj

    def forward(self, input: Any, output: Any) -> None:
        """Execute the forward transform."""
        self._backend.forward(input, output)

    def backward(self, input: Any, output: Any) -> None:
        """Execute the backward transform."""
        self._backend.backward(input, output)

    def free(self) -> None:
        """Explicitly release C++ resources."""
        self._backend = None

    def __enter__(self) -> 'FFTPlan':
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.free()


# Factory function
def FFT(
    ndim: Optional[Union[int, Sequence[int]]] = None,
    shape: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    dtype: str = "float64",
    backend: str = "fftw",
    comm: Optional[Any] = None,
    grid: Optional[Sequence[int]] = None,
    threads: int = 1,
    flags: Optional[int] = None,
) -> FFTPlan:
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
        threads (int, optional): Number of OpenMP threads to use (fftw backend only). Defaults to 1.
        flags (int, optional): FFTW planner flags (e.g., anyFFT.FFTW_MEASURE). Defaults to anyFFT.FFTW_ESTIMATE. (fftw and fftw_mpi backend only).

    Returns:
        An instance of the requested FFT backend.
    """

    comm_handle = 0
    if comm is not None:
        # Handle mpi4py object -> Fortran Integer Handle conversion
        if hasattr(comm, "py2f"):
            comm_handle = comm.py2f()
        elif isinstance(comm, int):
            comm_handle = comm
        else:
            raise TypeError("Communicator must be an mpi4py communicator or an integer handle.")

    if grid is None:
        grid = []


    if isinstance(ndim, (list, tuple)):
        shape = ndim

    if not isinstance(shape, (list, tuple)):
        raise ValueError("Must provide valid 'shape' to create a plan.")

    ndim = len(shape)

    if axes is None:
        axes = []

    if not isinstance(dtype, str):
        if hasattr(dtype, "name"):
            dtype = dtype.name  # np.dtype(np.float64).name -> 'float64'
        else:
            dtype = str(dtype).replace("'", "").replace("<class ", "").replace(">", "").split(".")[-1]

    dtype = dtype.lower()
    if dtype not in ["float32", "float64", "complex64", "complex128"]:
        raise ValueError("Invalid 'dtype' specified. Must be one of: 'float32', 'float64', 'complex64', 'complex128'.")


    backend_obj = None
    if backend == "fftw":
        if flags is None:
            flags = FFTW_ESTIMATE

        if comm is None:
            if not _HAS_FFTW:
                raise RuntimeError("anyFFT was compiled without FFTW support.")

            if input is None or output is None:
                raise ValueError("The 'fftw' backend requires 'input' and 'output' dummy arrays to create a plan.")

            backend_obj = fftw_serial(shape, axes, input, output, dtype, threads, flags)

        else:
            if not _HAS_FFTW_MPI:
                raise RuntimeError("anyFFT was compiled without FFTW-MPI support.")

            if input is None or output is None:
                raise ValueError("The 'fftw_mpi' backend requires 'input' and 'output' (local) arrays to create a plan.")

            backend_obj = fftw_mpi(shape, input, output, dtype, comm_handle)

    elif backend == "cufft":
        if comm is None:
            if not _HAS_CUDA:
                raise RuntimeError("anyFFT was compiled without CUDA support.")

            backend_obj = cufft_serial(shape, axes, dtype)

        else:
            if not _HAS_CUDA_MPI:
                raise RuntimeError("anyFFT was compiled without CUDA support.")

            if input is None or output is None:
                raise ValueError("The 'cufft_mpi' backend requires 'input' and 'output' arrays to check for In-Place/Out-of-Place consistency.")

            backend_obj = cufft_mpi(shape, grid, input, output, dtype, comm_handle)

    else:
        raise ValueError(f"Unknown backend type: '{backend}'. Available: {__backends__}")

    return FFTPlan(backend_obj)


# Functional Helpers
def fft(a: Any, out: Optional[Any] = None, axes: Optional[Sequence[int]] = None, backend: str = "fftw", comm: Optional[Any] = None) -> Any:
    """Perform a one-off forward FFT."""
    if out is None:
        raise ValueError("anyFFT requires explicit 'out' array for functional API to determine types.")

    shape = a.shape
    dtype = a.dtype

    plan = FFT(shape=shape, axes=axes, input=a, output=out, dtype=dtype, backend=backend, comm=comm)
    plan.forward(a, out)
    return out

def ifft(a: Any, out: Optional[Any] = None, axes: Optional[Sequence[int]] = None, backend: str = "fftw", comm: Optional[Any] = None) -> Any:
    """Perform a one-off backward FFT."""
    if out is None:
        raise ValueError("anyFFT requires explicit 'out' array for functional API to determine types.")

    shape = out.shape
    dtype = out.dtype

    plan = FFT(shape=shape, axes=axes, input=out, output=a, dtype=dtype, backend=backend, comm=comm)
    plan.backward(a, out)
    return out
