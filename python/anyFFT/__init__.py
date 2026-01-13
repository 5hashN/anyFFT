_HAS_FFTW = False
_HAS_CUDA = False

try:
    from .anyFFT import fftw
    _HAS_FFTW = True
except ImportError:
    pass  # FFTW backend not compiled

try:
    from .anyFFT import cufft
    _HAS_CUDA = True
except ImportError:
    pass  # CUDA backend not compiled

# Define Exports
__all__ = ["FFT"]
if _HAS_FFTW: __all__.append("fftw")
if _HAS_CUDA: __all__.append("cufft")

# Factory function
def FFT(ndim, shape, input=None, output=None, dtype="float64", backend="fftw"):
    """
    Factory function to create an FFT backend instance.

    Args:
        ndim (int): Number of dimensions.
        shape (list): Dimensions of the transform.
        input (array, optional): Dummy input array (Required for FFTW).
        output (array, optional): Dummy output array (Required for FFTW).
        dtype (str): 'float32' or 'float64'.
        backend (str): 'fftw' or 'cufft'.
    """

    if backend == "fftw":
        if not _HAS_FFTW:
            raise RuntimeError("The 'fftw' backend was requested, but anyFFT was compiled without FFTW support.")

        if input is None or output is None:
            raise ValueError("FFTW backend requires 'input' and 'output' dummy arrays to create a plan.")

        return fftw(ndim, shape, input, output, dtype)

    elif backend == "cufft":
        if not _HAS_CUDA:
            raise RuntimeError("The 'cufft' backend was requested, but anyFFT was compiled without CUDA support.")

        return cufft(ndim, shape, dtype)

    else:
        raise ValueError(f"Unknown backend type: '{backend}'. Available: 'fftw', 'cufft'")