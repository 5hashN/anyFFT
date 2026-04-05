"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import pytest

import numpy as np
from .. import test_utils
import anyFFT
from anyFFT import FFT

if not anyFFT.has_backend("fftw"):
    pytest.skip("Skipping fftw tests: Backend not compiled", allow_module_level=True)

BACKEND = "fftw"

SCENARIOS = [
    ([1024], []),
    ([128, 64], [0]),
    ([128, 64], [1]),
    ([128, 64], []),
    ([16, 32, 32], [0, 1]),
    ([16, 32, 32], [0, 1, 2]),
    ([16, 32, 32], []),
    ([4, 8, 32, 32], [2, 3]),
    ([4, 8, 32, 32], [0, 1, 2, 3]),
]

def setup_r2c_inplace_buffer(shape, dtype_str, axes=None):
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)
    complex_shape = list(shape)
    target_axis = axes[-1] if axes else -1
    complex_shape[target_axis] = complex_shape[target_axis] // 2 + 1

    buffer_complex = np.zeros(complex_shape, dtype=complex_dtype)
    buffer_real_view = buffer_complex.view(real_dtype)

    slices = [slice(None)] * len(shape)
    slices[target_axis] = slice(0, shape[target_axis])
    return buffer_complex, buffer_real_view[tuple(slices)]

@pytest.mark.cpu
@pytest.mark.parametrize(
    "shape,axes",
    SCENARIOS,
    ids=[f"shape={s}_axes={a}" for s, a in SCENARIOS]
)
@pytest.mark.parametrize("dtype_str", ["float64", "float32"])
@pytest.mark.parametrize("is_r2c", [False, True])
@pytest.mark.parametrize("is_inplace", [False, True])
def test_fftw_local(shape, axes, dtype_str, is_r2c, is_inplace):
    if is_r2c and is_inplace and axes and ((len(shape) - 1) not in axes):
        pytest.skip("R2C inplace requires the last dimension to be transformed.")

    full_dtype_str = dtype_str if is_r2c else test_utils.get_c2c_dtype_str(dtype_str)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    if not is_r2c:
        in_data = test_utils.generate_data(shape, complex_dtype)
        if is_inplace:
            in_buffer, out_buffer = np.copy(in_data), None
            out_buffer = in_buffer
        else:
            in_buffer, out_buffer = in_data, np.zeros_like(in_data)
        copy_ref = in_data
    else:
        clean_real_input = test_utils.generate_data(shape, real_dtype)
        if is_inplace:
            buffer_complex, input_real_slice = setup_r2c_inplace_buffer(shape, dtype_str, axes)
            np.copyto(input_real_slice, clean_real_input)
            in_buffer, out_buffer = input_real_slice, buffer_complex
        else:
            out_shape = list(shape)
            target_ax = axes[-1] if axes else -1
            out_shape[target_ax] = out_shape[target_ax] // 2 + 1
            in_buffer = clean_real_input
            out_buffer = np.zeros(out_shape, dtype=complex_dtype)
        copy_ref = clean_real_input

    try:
        fft = FFT(shape=shape, axes=axes, input=in_buffer, output=out_buffer, dtype=full_dtype_str, backend=BACKEND)
        fft.forward(in_buffer, out_buffer)

        if is_inplace:
            fft.backward(out_buffer, in_buffer if is_r2c else out_buffer)
            result_data = in_buffer if is_r2c else out_buffer
        else:
            result_data = np.zeros_like(in_buffer)
            fft.backward(out_buffer, result_data)

    except NotImplementedError as e:
        pytest.skip(f"Backend limitation: {str(e)}")

    tol = 1e-4 if "float32" in dtype_str else 1e-12
    np.testing.assert_allclose(result_data, copy_ref, atol=tol, rtol=0, err_msg="FFT roundtrip failed tolerance check")
