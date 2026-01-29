import numpy as np
import cupy as cp
import sys
import test_utils

try:
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

BACKEND = "cufft"

def test_r2c_out_of_place(dtype_str, shape, ndim):
    test_utils.print_test_start("[R2C Out-Place]", dtype_str)

    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    # Get numpy types for host generation
    np_real, _ = test_utils.get_numpy_types(dtype_str)

    shape_f = list(shape)
    shape_f[-1] = shape_f[-1] // 2 + 1

    dummy_real = cp.empty(shape, dtype=real_dtype)
    dummy_complex = cp.empty(shape_f, dtype=complex_dtype)

    try:
        fft = FFT(ndim, shape, input=dummy_real, output=dummy_complex, dtype=dtype_str, backend=BACKEND)
    except Exception as e:
        test_utils.print_test_result(0, error=e)
        return

    # Generate on Host -> Transfer to Device
    real_in_host = test_utils.generate_data(shape, np_real)
    real_in = cp.asarray(real_in_host)

    complex_out = cp.empty(shape_f, dtype=complex_dtype)
    real_back = cp.empty(shape, dtype=real_dtype)

    fft.forward(real_in, complex_out)
    fft.backward(complex_out, real_back)

    diff = cp.max(cp.abs(real_in - real_back)).item()
    test_utils.print_test_result(diff)

def test_r2c_in_place(dtype_str, shape, ndim):
    test_utils.print_test_start("[R2C In-Place ]", dtype_str)

    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    np_real, _ = test_utils.get_numpy_types(dtype_str)

    shape_f = list(shape)
    shape_f[-1] = shape_f[-1] // 2 + 1

    data_buffer = cp.zeros(shape_f, dtype=complex_dtype)
    real_view = data_buffer.view(real_dtype)
    valid_n = shape[-1]

    # Host -> Device
    clean_input_host = test_utils.generate_data(shape, np_real)
    clean_input = cp.asarray(clean_input_host)

    real_view[..., :valid_n] = clean_input

    try:
        fft = FFT(ndim, shape, input=data_buffer, output=data_buffer, dtype=dtype_str, backend=BACKEND)
    except Exception as e:
        test_utils.print_test_result(0, error=e)
        return

    fft.forward(data_buffer, data_buffer)
    fft.backward(data_buffer, data_buffer)

    result = real_view[..., :valid_n]
    diff = cp.max(cp.abs(clean_input - result)).item()
    test_utils.print_test_result(diff)

def test_c2c_out_of_place(real_dtype_str, shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C Out-Place]", c_dtype_str)

    _, complex_dtype = test_utils.get_cupy_types(real_dtype_str)
    _, np_complex = test_utils.get_numpy_types(real_dtype_str)

    in_data_host = test_utils.generate_data(shape, np_complex)
    in_data = cp.asarray(in_data_host)

    out_data = cp.empty_like(in_data)
    back_data = cp.empty_like(in_data)

    try:
        fft = FFT(ndim, shape, input=in_data, output=out_data, dtype=c_dtype_str, backend=BACKEND)
    except Exception as e:
        test_utils.print_test_result(0, error=e)
        return

    fft.forward(in_data, out_data)
    fft.backward(out_data, back_data)

    diff = cp.max(cp.abs(in_data - back_data)).item()
    test_utils.print_test_result(diff)

def test_c2c_in_place(real_dtype_str, shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C In-Place ]", c_dtype_str)

    _, complex_dtype = test_utils.get_cupy_types(real_dtype_str)
    _, np_complex = test_utils.get_numpy_types(real_dtype_str)

    clean_input_host = test_utils.generate_data(shape, np_complex)
    clean_input = cp.asarray(clean_input_host)

    data_buffer = cp.copy(clean_input)

    try:
        fft = FFT(ndim, shape, input=data_buffer, output=data_buffer, dtype=c_dtype_str, backend=BACKEND)
    except Exception as e:
        test_utils.print_test_result(0, error=e)
        return

    fft.forward(data_buffer, data_buffer)
    fft.backward(data_buffer, data_buffer)

    diff = cp.max(cp.abs(clean_input - data_buffer)).item()
    test_utils.print_test_result(diff)

def main():
    test_utils.print_header(BACKEND)
    precisions = ["float64", "float32"]
    configurations = [(2, [1024, 1024]), (3, [128, 128, 128])]

    for ndim, shape in configurations:
        test_utils.print_config(ndim, shape)
        for dtype in precisions:
            print(f"Precision: {dtype}")
            test_r2c_out_of_place(dtype, shape, ndim)
            test_r2c_in_place(dtype, shape, ndim)
            test_c2c_out_of_place(dtype, shape, ndim)
            test_c2c_in_place(dtype, shape, ndim)
            print("")
    print("Tests Completed.")

if __name__ == "__main__":
    main()