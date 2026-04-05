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
from anyFFT import FFT, get_gpu_backend_name

if not anyFFT.has_backend("gpufft"):
    pytest.skip("Skipping gpufft tests: Backend not compiled", allow_module_level=True)

cp = pytest.importorskip("cupy")
BACKEND = get_gpu_backend_name()

SCENARIOS = [
    ([1024], []),
    ([128, 64], [0]),
    ([128, 64], [1]),
    ([128, 64], []),
    ([16, 32, 32], [0, 1]),
    ([16, 32, 32], [0, 1, 2]),
    ([16, 32, 32], [])
]

def setup_r2c_inplace_buffer_gpu(shape, dtype_str, axes=None):
    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    complex_shape = list(shape)
    target_axis = axes[-1] if axes else -1
    complex_shape[target_axis] = complex_shape[target_axis] // 2 + 1

    buffer_complex = cp.zeros(complex_shape, dtype=complex_dtype)
    buffer_real_view = buffer_complex.view(real_dtype)

    slices = [slice(None)] * len(shape)
    slices[target_axis] = slice(0, shape[target_axis])
    return buffer_complex, buffer_real_view[tuple(slices)]

@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,axes",
    SCENARIOS,
    ids=[f"shape={s}_axes={a}" for s, a in SCENARIOS]
)
@pytest.mark.parametrize("dtype_str", ["float64", "float32"])
@pytest.mark.parametrize("is_r2c", [False, True])
@pytest.mark.parametrize("is_inplace", [False, True])
def test_gpufft_local(shape, axes, dtype_str, is_r2c, is_inplace):
    if is_r2c and is_inplace and axes and ((len(shape) - 1) not in axes):
        pytest.skip("R2C inplace requires the last dimension to be transformed.")

    full_dtype_str = dtype_str if is_r2c else test_utils.get_c2c_dtype_str(dtype_str)
    real_np, complex_np = test_utils.get_numpy_types(dtype_str)
    _, complex_cp = test_utils.get_cupy_types(dtype_str)

    host_data = test_utils.generate_data(shape, real_np if is_r2c else complex_np)
    ref_gpu = cp.asarray(host_data)

    if not is_r2c:
        if is_inplace:
            in_buffer, out_buffer = cp.copy(ref_gpu), None
            out_buffer = in_buffer
        else:
            in_buffer, out_buffer = ref_gpu, cp.zeros_like(ref_gpu)
    else:
        if is_inplace:
            buffer_complex, input_real_slice = setup_r2c_inplace_buffer_gpu(shape, dtype_str, axes)
            input_real_slice[:] = ref_gpu
            in_buffer, out_buffer = input_real_slice, buffer_complex
        else:
            out_shape = list(shape)
            target_ax = axes[-1] if axes else -1
            out_shape[target_ax] = out_shape[target_ax] // 2 + 1
            in_buffer = ref_gpu
            out_buffer = cp.zeros(out_shape, dtype=complex_cp)

    try:
        fft = FFT(shape=tuple(shape), axes=tuple(axes), input=in_buffer, output=out_buffer, dtype=full_dtype_str, backend=BACKEND)
        fft.forward(in_buffer, out_buffer)

        if is_inplace:
            fft.backward(out_buffer, in_buffer if is_r2c else out_buffer)
            result_gpu = in_buffer if is_r2c else out_buffer
        else:
            result_gpu = cp.zeros_like(in_buffer)
            fft.backward(out_buffer, result_gpu)

    except NotImplementedError as e:
        pytest.skip(f"Backend limitation: {str(e)}")

    tol = 1e-4 if "float32" in dtype_str else 1e-12
    diff = cp.max(cp.abs(ref_gpu - result_gpu)).item()
    assert diff < tol, f"Max difference {diff:.2e} exceeded tolerance {tol}"
