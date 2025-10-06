import numpy as np
from anyFFT import FFT

shape = [128, 128, 128]
shapeF = [128, 128, 65]
ndim = 3
dtype = "float64"  # or "float64"
backend = "fftw"

if dtype == "float32":
    dummy_real = np.empty(shape, dtype=np.float32)
    dummy_complex = np.empty(shapeF, dtype=np.complex64)
else:
    dummy_real = np.empty(shape, dtype=np.float64)
    dummy_complex = np.empty(shapeF, dtype=np.complex128)

fft = FFT(ndim, shape, dummy_real, dummy_complex, dtype, backend)

# Actual computation
real_in = np.random.rand(*shape).astype(dummy_real.dtype)

complex_out = np.empty(shapeF, dtype=dummy_complex.dtype)
fft.forward(real_in, complex_out)

real_out = np.empty(shape, dtype=dummy_real.dtype)
fft.backward(complex_out, real_out)

print("Max diff:", np.max(np.abs(real_in - real_out)))