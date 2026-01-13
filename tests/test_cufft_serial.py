import numpy as np
import cupy as cp
from anyFFT import FFT

shape = [128, 128, 128]
shapeF = [128, 128, 65]
ndim = 3
dtype = "float64"
backend = "cufft"

if dtype == "float32":
    real_dtype = cp.float32
    complex_dtype = cp.complex64
else:
    real_dtype = cp.float64
    complex_dtype = cp.complex128

fft = FFT(ndim, shape, dtype=dtype, backend=backend)

real_in = cp.random.rand(*shape).astype(real_dtype)

complex_out = cp.empty(shapeF, dtype=complex_dtype)
fft.forward(real_in, complex_out)

real_out = cp.empty(shape, dtype=real_dtype)
fft.backward(complex_out, real_out)

print("Max diff:", cp.max(cp.abs(real_in - real_out)).item())