_HAS_FFTW = False
_HAS_FFTW_MPI = False
_HAS_CUDA = False
_HAS_CUDA_MPI = False

try:
    from .anyFFT import fftw
    _HAS_FFTW = True
except ImportError:
    pass  # FFTW Serial backend not compiled

try:
    from .anyFFT import fftw_mpi
    _HAS_FFTW_MPI = True
except ImportError:
    pass  # FFTW MPI backend not compiled

try:
    from .anyFFT import cufft
    _HAS_CUDA = True
except ImportError:
    pass  # CUDA backend not compiled

try:
    from .anyFFT import cufft_mpi
    _HAS_CUDA_MPI = True
except ImportError:
    pass # CUDA MPI backend not compiled

# Define Exports
__all__ = ["FFT"]
if _HAS_FFTW: __all__.append("fftw")
if _HAS_FFTW_MPI: __all__.append("fftw_mpi")
if _HAS_CUDA: __all__.append("cufft")
if _HAS_CUDA_MPI: __all__.append("cufft_mpi")

def has_backend(backend_name):
    """
    Checks if a specific backend is compiled and available.

    Args:
        backend_name (str): 'fftw', 'fftw_mpi', 'cufft', or 'cufft_mpi'.

    Returns:
        bool: True if available, False otherwise.
    """
    if backend_name == "fftw": return _HAS_FFTW
    if backend_name == "fftw_mpi": return _HAS_FFTW_MPI
    if backend_name == "cufft": return _HAS_CUDA
    if backend_name == "cufft_mpi": return _HAS_CUDA_MPI
    return False

# Factory function
def FFT(ndim, shape, input=None, output=None, comm=None, dtype="float64", backend="fftw"):
    """
    Factory function to create an FFT backend instance.

    Args:
        ndim (int): Number of dimensions (e.g., 2 or 3).
        shape (list/tuple): Dimensions of the transform.
        input (array, optional): Input array (Required for FFTW planning).
        output (array, optional): Output array (Required for FFTW planning).
        comm (mpi4py.MPI.Comm, optional): MPI Communicator (Required for MPI backends).
        dtype (str): 'float32', 'float64', 'complex64', or 'complex128'.
        backend (str): 'fftw', 'fftw_mpi', 'cufft', or 'cufft_mpi'.

    Returns:
        An instance of the requested FFT backend.
    """

    if backend == "fftw":
        if not _HAS_FFTW:
            raise RuntimeError("The 'fftw' backend was requested, but anyFFT was compiled without FFTW support.")

        if input is None or output is None:
            raise ValueError("FFTW backend requires 'input' and 'output' dummy arrays to create a plan.")

        return fftw(ndim, shape, input, output, dtype)

    elif backend == "fftw_mpi":
        if not _HAS_FFTW_MPI:
            raise RuntimeError("The 'fftw_mpi' backend was requested, but anyFFT was compiled without FFTW MPI support.")

        if comm is None:
            raise ValueError("FFTW MPI backend requires 'comm' (an mpi4py communicator).")

        if input is None or output is None:
            raise ValueError("FFTW MPI backend requires 'input' and 'output' (local) arrays to create a plan.")

        # Handle mpi4py object -> Fortran Integer Handle conversion
        comm_handle = comm
        if hasattr(comm, "py2f"):
            comm_handle = comm.py2f()

        # Ensure comm is an integer
        if not isinstance(comm_handle, int):
             raise TypeError("Communicator must be an mpi4py communicator or an integer handle.")

        return fftw_mpi(ndim, shape, input, output, comm_handle, dtype)

    elif backend == "cufft":
        if not _HAS_CUDA:
            raise RuntimeError("The 'cufft' backend was requested, but anyFFT was compiled without CUDA support.")

        return cufft(ndim, shape, dtype)

    elif backend == "cufft_mpi":
        if not _HAS_CUDA:
            raise RuntimeError("The 'cufft_mpi' backend was requested, but anyFFT was compiled without CUDA support.")

        if comm is None:
            raise ValueError("CuFFT MPI backend requires 'comm' (an mpi4py communicator).")

        # Handle mpi4py object -> Fortran Integer Handle conversion
        comm_handle = comm
        if hasattr(comm, "py2f"):
            comm_handle = comm.py2f()

        # Ensure comm is an integer
        if not isinstance(comm_handle, int):
             raise TypeError("Communicator must be an mpi4py communicator or an integer handle.")

        return cufft_mpi(ndim, shape, input, output, comm_handle, dtype)

    else:
        raise ValueError(f"Unknown backend type: '{backend}'. Available: {__all__[1:]}")