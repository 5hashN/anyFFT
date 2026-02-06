import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_utils

try:
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

BACKEND = "fftw"
FAILED_TESTS = []
TOTAL_TESTS = 0
PASSED_TESTS = 0

def setup_r2c_inplace_buffer(shape, dtype_str, axes=None):
    """
    Helper to create a padded buffer for In-Place R2C transforms.
    """
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    complex_shape = list(shape)
    target_axis = axes[-1] if axes else -1
    complex_shape[target_axis] = complex_shape[target_axis] // 2 + 1

    buffer_complex = np.zeros(complex_shape, dtype=complex_dtype)
    buffer_real_view = buffer_complex.view(real_dtype)

    slices = [slice(None)] * len(shape)
    slices[target_axis] = slice(0, shape[target_axis])
    input_real_slice = buffer_real_view[tuple(slices)]

    return buffer_complex, input_real_slice

def run_fftw_serial_test(label, shape, dtype_str, axes=None, ndim=None, is_r2c=False, is_inplace=False):
    """
    Unified test runner. Records failures to global list.
    """
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1

    if is_r2c:
        test_type = "R2C"
        full_dtype_str = dtype_str
        real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)
    else:
        test_type = "C2C"
        full_dtype_str = test_utils.get_c2c_dtype_str(dtype_str)
        _, complex_dtype = test_utils.get_numpy_types(dtype_str)

    place_str = "In-Place" if is_inplace else "Out-Place"
    config_str = f"axes={axes}" if axes else f"ndim={ndim}"

    test_id = f"{label} {test_type} {place_str} | {config_str} | {shape} | {dtype_str}"
    test_utils.print_test_start(f"[{test_type} {place_str}] {config_str}", full_dtype_str)

    try:
        if not is_r2c:
            in_data = test_utils.generate_data(shape, complex_dtype)
            if is_inplace:
                in_buffer = np.copy(in_data)
                out_buffer = in_buffer
                copy_ref = np.copy(in_data)
            else:
                in_buffer = in_data
                out_buffer = np.zeros_like(in_data)
                copy_ref = in_data
        else:
            clean_real_input = test_utils.generate_data(shape, real_dtype)
            if is_inplace:
                buffer_complex, input_real_slice = setup_r2c_inplace_buffer(shape, dtype_str, axes)
                np.copyto(input_real_slice, clean_real_input)
                in_buffer = input_real_slice
                out_buffer = buffer_complex
                copy_ref = clean_real_input
            else:
                out_shape = list(shape)
                target_ax = axes[-1] if axes else -1
                out_shape[target_ax] = out_shape[target_ax] // 2 + 1
                in_buffer = clean_real_input
                out_buffer = np.zeros(out_shape, dtype=complex_dtype)
                copy_ref = clean_real_input

        fft = FFT(ndim=ndim, shape=shape, axes=axes,
                  input=in_buffer, output=out_buffer,
                  dtype=full_dtype_str, backend=BACKEND)

        fft.forward(in_buffer, out_buffer)

        if is_r2c:
            if is_inplace:
                fft.backward(out_buffer, in_buffer)
                result_data = in_buffer
            else:
                back_buffer = np.zeros_like(in_buffer)
                fft.backward(out_buffer, back_buffer)
                result_data = back_buffer
        else:
            if is_inplace:
                fft.backward(out_buffer, out_buffer)
                result_data = out_buffer
            else:
                back_buffer = np.zeros_like(in_buffer)
                fft.backward(out_buffer, back_buffer)
                result_data = back_buffer

        diff = np.max(np.abs(copy_ref - result_data))

        tol = 1e-4 if "float32" in dtype_str or "complex64" in dtype_str else 1e-12

        if diff < tol:
            test_utils.print_test_result(diff, tol=tol)
            PASSED_TESTS += 1
        else:
            test_utils.print_test_result(diff, tol=tol)
            FAILED_TESTS.append(f"{test_id} -> Failed (Diff: {diff:.2e})")
    except Exception as e:
        test_utils.print_test_result(0, error=e)
        FAILED_TESTS.append(f"{test_id} -> Error: {str(e)}")

def main():
    test_utils.print_header(BACKEND)
    # Hardcoded Suite
    print(f"\n{'='*60}\n  HARDCODED \n{'='*60}")
    for ndim, shape in [(1, [2048]), (2, [1024, 1024]), (3, [64, 64, 64])]:
        test_utils.print_config(ndim, shape)
        for dtype in ["float64", "float32"]:
            for r2c in [False, True]:
                for inplace in [False, True]:
                    run_fftw_serial_test("HC", shape, dtype, axes=None, ndim=ndim, is_r2c=r2c, is_inplace=inplace)

    # Generic Suite
    print(f"\n{'='*60}\n  GENERIC \n{'='*60}")
    scenarios = [
        { "shape": [1024], "axes": [0], "desc": "1D Full" },
        { "shape": [128, 64], "axes": [0], "desc": "2D Axis 0" },
        { "shape": [128, 64], "axes": [1], "desc": "2D Axis 1" },
        { "shape": [128, 64], "axes": [0, 1], "desc": "2D Full" },
        { "shape": [16, 32, 32], "axes": [0, 1, 2], "desc": "3D Full" },
        { "shape": [4, 8, 32, 32], "axes": [2, 3],       "desc": "4D Spatial" },
        { "shape": [4, 8, 32, 32], "axes": [0, 1, 2, 3], "desc": "4D Full" },
    ]

    for sc in scenarios:
        print(f"\n--- {sc['desc']} ---")
        for dtype in ["float64", "float32"]:
            run_fftw_serial_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=False, is_inplace=False)
            run_fftw_serial_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=False, is_inplace=True)
            run_fftw_serial_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=True, is_inplace=False)
            if (len(sc["shape"])-1) in sc["axes"]:
                run_fftw_serial_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=True, is_inplace=True)

    print(f"\n{'='*60}\nSUMMARY: Total {TOTAL_TESTS} | Passed {PASSED_TESTS} | Failed {len(FAILED_TESTS)}")
    if FAILED_TESTS: sys.exit(1)

if __name__ == "__main__":
    main()