import numpy as np
import sys

# Try to import the library
try:
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'. Make sure the library is installed.")
    sys.exit(1)

# Configuration
BACKEND = "fftw"  # or "cufft"

def get_c2c_dtype_str(real_dtype_str):
    """Maps 'float64' -> 'complex128' for the library constructor."""
    if real_dtype_str == "float64":
        return "complex128"
    elif real_dtype_str == "float32":
        return "complex64"
    else:
        raise ValueError("Unknown precision")

def get_numpy_types(dtype_str):
    """Returns (real_dtype, complex_dtype) based on the string."""
    if dtype_str == "float32" or dtype_str == "complex64":
        return np.float32, np.complex64
    elif dtype_str == "float64" or dtype_str == "complex128":
        return np.float64, np.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

# R2C Tests
def test_r2c_out_of_place(dtype_str, shape, ndim):
    print(f"  [R2C Out-Place] Testing {dtype_str}...", end=" ")

    real_dtype, complex_dtype = get_numpy_types(dtype_str)

    # Output Shape (Last dim is N//2 + 1)
    shape_f = list(shape)
    shape_f[-1] = shape_f[-1] // 2 + 1

    # Alloc
    dummy_real = np.empty(shape, dtype=real_dtype)
    dummy_complex = np.empty(shape_f, dtype=complex_dtype)

    # Init
    try:
        fft = FFT(ndim, shape, input=dummy_real, output=dummy_complex, dtype=dtype_str, backend=BACKEND)
    except Exception as e:
        print(f"FAILED (Init)\n    {e}")
        return

    # Data
    real_in = np.random.rand(*shape).astype(real_dtype)
    complex_out = np.empty(shape_f, dtype=complex_dtype)
    real_back = np.empty(shape, dtype=real_dtype)

    # Exec
    fft.forward(real_in, complex_out)
    fft.backward(complex_out, real_back)

    # Verify
    diff = np.max(np.abs(real_in - real_back))
    if diff < 1e-5: print(f"PASSED (Diff: {diff:.2e})")
    else: print(f"FAILED (Diff: {diff:.2e})")

def test_r2c_in_place(dtype_str, shape, ndim):
    print(f"  [R2C In-Place ] Testing {dtype_str}...", end=" ")

    real_dtype, complex_dtype = get_numpy_types(dtype_str)

    # Shape (Complex Padded)
    shape_f = list(shape)
    shape_f[-1] = shape_f[-1] // 2 + 1

    # Alloc ONE buffer
    data_buffer = np.zeros(shape_f, dtype=complex_dtype)
    real_view = data_buffer.view(real_dtype)

    # Fill Valid Data
    valid_n = shape[-1]
    clean_input = np.random.rand(*shape).astype(real_dtype)
    real_view[..., :valid_n] = clean_input

    # Init
    try:
        fft = FFT(ndim, shape, input=data_buffer, output=data_buffer, dtype=dtype_str, backend=BACKEND)
    except Exception as e:
        print(f"FAILED (Init)\n    {e}")
        return

    # Exec
    fft.forward(data_buffer, data_buffer)
    fft.backward(data_buffer, data_buffer)

    # Verify
    result = real_view[..., :valid_n]
    diff = np.max(np.abs(clean_input - result))
    if diff < 1e-5: print(f"PASSED (Diff: {diff:.2e})")
    else: print(f"FAILED (Diff: {diff:.2e})")

# C2C Tests
def test_c2c_out_of_place(real_dtype_str, shape, ndim):
    c_dtype_str = get_c2c_dtype_str(real_dtype_str)
    print(f"  [C2C Out-Place] Testing {c_dtype_str}...", end=" ")

    _, complex_dtype = get_numpy_types(real_dtype_str)

    # Data (Complex input, same shape out)
    in_data = np.random.rand(*shape).astype(complex_dtype) + \
              1j * np.random.rand(*shape).astype(complex_dtype)
    out_data = np.empty_like(in_data)
    back_data = np.empty_like(in_data)

    # Init
    try:
        fft = FFT(ndim, shape, input=in_data, output=out_data, dtype=c_dtype_str, backend=BACKEND)
    except Exception as e:
        print(f"FAILED (Init)\n    {e}")
        return

    # Exec
    fft.forward(in_data, out_data)
    fft.backward(out_data, back_data)

    # Verify
    diff = np.max(np.abs(in_data - back_data))
    if diff < 1e-5: print(f"PASSED (Diff: {diff:.2e})")
    else: print(f"FAILED (Diff: {diff:.2e})")

def test_c2c_in_place(real_dtype_str, shape, ndim):
    c_dtype_str = get_c2c_dtype_str(real_dtype_str)
    print(f"  [C2C In-Place ] Testing {c_dtype_str}...", end=" ")

    _, complex_dtype = get_numpy_types(real_dtype_str)

    # Data
    clean_input = np.random.rand(*shape).astype(complex_dtype) + \
                  1j * np.random.rand(*shape).astype(complex_dtype)

    # Buffer that will be overwritten
    data_buffer = np.copy(clean_input)

    # Init (Input == Output)
    try:
        fft = FFT(ndim, shape, input=data_buffer, output=data_buffer, dtype=c_dtype_str, backend=BACKEND)
    except Exception as e:
        print(f"FAILED (Init)\n    {e}")
        return

    # Exec
    fft.forward(data_buffer, data_buffer)
    fft.backward(data_buffer, data_buffer)

    # Verify
    diff = np.max(np.abs(clean_input - data_buffer))
    if diff < 1e-5: print(f"PASSED (Diff: {diff:.2e})")
    else: print(f"FAILED (Diff: {diff:.2e})")


def main():
    print(f"========================================")
    print(f"Testing anyFFT Backend: {BACKEND}")
    print(f"========================================\n")

    precisions = ["float64", "float32"]

    # Define test configurations (NDIM, SHAPE)
    configurations = [
        (2, [1024, 1024]),
        (3, [128, 128, 128])
    ]

    for ndim, shape in configurations:
        print(f"----------------------------------------")
        print(f"Config: {ndim}D | Shape: {shape}")
        print(f"----------------------------------------")

        for dtype in precisions:
            print(f"Precision: {dtype}")

            # Test Real-to-Complex
            test_r2c_out_of_place(dtype, shape, ndim)
            test_r2c_in_place(dtype, shape, ndim)

            # Test Complex-to-Complex (Mapped)
            test_c2c_out_of_place(dtype, shape, ndim)
            test_c2c_in_place(dtype, shape, ndim)

            print("")

    print("========================================")
    print("Tests Completed.")

if __name__ == "__main__":
    main()