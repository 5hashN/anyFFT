import numpy as np
import sys
import test_utils

# Import CuPy
try:
    import cupy as cp
except ImportError:
    print("Error: CuPy not installed.")
    sys.exit(1)

try:
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

BACKEND = "cufft"

FAILED_TESTS = []
TOTAL_TESTS = 0
PASSED_TESTS = 0

def setup_r2c_inplace_buffer_gpu(shape, dtype_str, axes=None):
    """
    Helper to create a padded GPU buffer for In-Place R2C transforms.
    Allocates the Complex (output) size, then views it as Real to create the Input.
    """
    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)

    complex_shape = list(shape)
    target_axis = axes[-1] if axes else -1
    complex_shape[target_axis] = complex_shape[target_axis] // 2 + 1

    buffer_complex = cp.zeros(complex_shape, dtype=complex_dtype)

    buffer_real_view = buffer_complex.view(real_dtype)
    slices = [slice(None)] * len(shape)
    slices[target_axis] = slice(0, shape[target_axis])

    input_real_slice = buffer_real_view[tuple(slices)]

    return buffer_complex, input_real_slice

def run_cufft_test(label, shape, dtype_str, axes=None, ndim=None, is_r2c=False, is_inplace=False):
    """
    Unified CuPy test runner. Records failures to global list.
    """
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1

    if is_r2c:
        test_type = "R2C"
        full_dtype_str = dtype_str
        real_dtype_np, _ = test_utils.get_numpy_types(dtype_str)
        _, complex_dtype_cp = test_utils.get_cupy_types(dtype_str)
    else:
        test_type = "C2C"
        full_dtype_str = test_utils.get_c2c_dtype_str(dtype_str)
        _, complex_dtype_np = test_utils.get_numpy_types(dtype_str)
        _, complex_dtype_cp = test_utils.get_cupy_types(dtype_str)

    place_str = "In-Place" if is_inplace else "Out-Place"
    config_str = f"axes={axes}" if axes else f"ndim={ndim}"

    test_id = f"{label} {test_type} {place_str} | {config_str} | {shape} | {dtype_str}"
    header_lbl = f"[{test_type} {place_str}] {config_str}"

    test_utils.print_test_start(header_lbl, full_dtype_str)

    try:
        if not is_r2c:
            host_data = test_utils.generate_data(shape, complex_dtype_np)
        else:
            host_data = test_utils.generate_data(shape, real_dtype_np)

        # Explicit transfer: ref_gpu is strictly a CuPy array.
        ref_gpu = cp.asarray(host_data)

        if not is_r2c:
            if is_inplace:
                in_buffer = cp.copy(ref_gpu)
                out_buffer = in_buffer
            else:
                in_buffer = ref_gpu
                out_buffer = cp.zeros_like(ref_gpu)
        else:
            if is_inplace:
                buffer_complex, input_real_slice = setup_r2c_inplace_buffer_gpu(shape, dtype_str, axes)
                input_real_slice[:] = ref_gpu
                in_buffer = input_real_slice
                out_buffer = buffer_complex
            else:
                out_shape = list(shape)
                target_ax = axes[-1] if axes else -1
                out_shape[target_ax] = out_shape[target_ax] // 2 + 1

                in_buffer = ref_gpu
                out_buffer = cp.zeros(out_shape, dtype=complex_dtype_cp)

        fft = FFT(ndim=len(shape), shape=tuple(shape), axes=tuple(axes) if axes else None,
                  input=in_buffer, output=out_buffer,
                  dtype=full_dtype_str, backend=BACKEND)

        fft.forward(in_buffer, out_buffer)

        if is_r2c:
            if is_inplace:
                fft.backward(out_buffer, in_buffer)
                result_gpu = in_buffer
            else:
                back_buffer = cp.zeros_like(in_buffer)
                fft.backward(out_buffer, back_buffer)
                result_gpu = back_buffer
        else:
            if is_inplace:
                fft.backward(out_buffer, out_buffer)
                result_gpu = out_buffer
            else:
                back_buffer = cp.zeros_like(in_buffer)
                fft.backward(out_buffer, back_buffer)
                result_gpu = back_buffer

        diff = cp.max(cp.abs(ref_gpu - result_gpu)).item()

        tol = 1e-4 if "float32" in dtype_str or "complex64" in dtype_str else 1e-12

        if diff < tol:
            test_utils.print_test_result(diff, tol=tol)
            PASSED_TESTS += 1
            return True
        else:
            test_utils.print_test_result(diff, tol=tol)
            FAILED_TESTS.append(f"{test_id} -> Failed (Diff: {diff:.2e} > {tol})")
            return False

    except Exception as e:
        test_utils.print_test_result(0, error=e)
        FAILED_TESTS.append(f"{test_id} -> Error: {str(e)}")
        return False

def test_legacy_suite():
    print(f"\n{'='*60}\n  HARDCODED \n{'='*60}")
    precisions = ["float64", "float32"]

    configurations = [
        (1, [2048]),
        (2, [1024, 1024]),
        (3, [64, 64, 64])
    ]

    for ndim, shape in configurations:
        test_utils.print_config(ndim, shape)
        for dtype in precisions:
            run_cufft_test("HC", shape, dtype, axes=None, ndim=ndim, is_r2c=False, is_inplace=False)
            run_cufft_test("HC", shape, dtype, axes=None, ndim=ndim, is_r2c=False, is_inplace=True)
            run_cufft_test("HC", shape, dtype, axes=None, ndim=ndim, is_r2c=True,  is_inplace=False)
            run_cufft_test("HC", shape, dtype, axes=None, ndim=ndim, is_r2c=True,  is_inplace=True)

def test_generic_suite():
    print(f"\n{'='*60}\n  GENERIC \n{'='*60}")
    precisions = ["float64", "float32"]

    scenarios = [
        { "shape": [1024], "axes": [0], "desc": "1D Full" },
        { "shape": [128, 64], "axes": [0],    "desc": "2D Axis 0 (Strided)" },
        { "shape": [128, 64], "axes": [1],    "desc": "2D Axis 1 (Contiguous)" },
        { "shape": [128, 64], "axes": [0, 1], "desc": "2D Full" },
        { "shape": [16, 32, 32], "axes": [0, 1, 2], "desc": "3D Full" },
        { "shape": [4, 8, 32, 32], "axes": [2, 3],       "desc": "4D Spatial" },
        { "shape": [4, 8, 32, 32], "axes": [0, 1, 2, 3], "desc": "4D Full" },
    ]

    for sc in scenarios:
        print(f"\n--- {sc['desc']} ---")
        for dtype in precisions:
            run_cufft_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=False, is_inplace=False)
            run_cufft_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=False, is_inplace=True)
            run_cufft_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=True, is_inplace=False)

            if (len(sc["shape"])-1) in sc["axes"]:
                run_cufft_test("Gen", sc["shape"], dtype, axes=sc["axes"], is_r2c=True, is_inplace=True)

def main():
    test_utils.print_header(BACKEND)

    test_legacy_suite()
    test_generic_suite()

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {TOTAL_TESTS}")
    print(f"Passed:      {PASSED_TESTS}")
    print(f"Failed:      {len(FAILED_TESTS)}")

    if FAILED_TESTS:
        print(f"\n[FAILED TESTS LIST]")
        for fail_msg in FAILED_TESTS:
            print(f"  - {fail_msg}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED.")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()