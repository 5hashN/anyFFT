"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import pytest

cp = pytest.importorskip("cupy")

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    pytest.skip("MPI library not functional on this system", allow_module_level=True)

import numpy as np
from .. import test_utils
import anyFFT
from anyFFT import FFT

if not anyFFT.has_backend("cufftmp"):
    pytest.skip("Skipping cufftmp tests: Backend not compiled", allow_module_level=True)

BACKEND = "cufftmp"

@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [[1024, 1024], [128, 128, 128]])
@pytest.mark.parametrize("dtype_str", ["float64", "float32"])
@pytest.mark.parametrize("is_r2c", [False, True])
@pytest.mark.parametrize("is_inplace", [False, True])
def test_cufftmp_distributed(shape, dtype_str, is_r2c, is_inplace):
    comm = MPI.COMM_WORLD
    ndim = len(shape)
    full_dtype_str = dtype_str if is_r2c else test_utils.get_c2c_dtype_str(dtype_str)
    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    np_real, np_complex = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = cufftmp.get_local_info(ndim, shape, comm.py2f(), is_r2c)
    except NotImplementedError as e:
        pytest.skip(str(e))

    if not is_r2c:
        in_buffer = cp.zeros(in_shape, dtype=complex_dtype)
        in_buffer[:] = cp.asarray(test_utils.generate_data(in_shape, np_complex, start_indices=in_start))

        if is_inplace:
            out_buffer = in_buffer
        else:
            out_buffer = cp.empty_like(in_buffer)
        ref_input = in_buffer.copy()

    else:
        if is_inplace:
            out_buffer = cp.zeros(out_shape, dtype=complex_dtype)
            real_view = out_buffer.view(real_dtype)
            valid_slice = real_view[:, :, :shape[2]] if ndim == 3 else real_view[:, :shape[1]]
            valid_slice[:] = cp.asarray(test_utils.generate_data(in_shape, np_real, start_indices=out_start))
            in_buffer = out_buffer
            ref_input = valid_slice.copy()
        else:
            in_buffer = cp.zeros(in_shape, dtype=real_dtype)
            in_buffer[:] = cp.asarray(test_utils.generate_data(in_shape, np_real, start_indices=in_start))
            out_buffer = cp.zeros(out_shape, dtype=complex_dtype)
            ref_input = in_buffer.copy()

    try:
        fft = FFT(ndim, shape, input=in_buffer, output=out_buffer, comm=comm, dtype=full_dtype_str, backend=BACKEND)
        fft.forward(in_buffer, out_buffer)

        if is_inplace:
            fft.backward(out_buffer, in_buffer)
            result = valid_slice if is_r2c else in_buffer
        else:
            result = cp.empty_like(in_buffer)
            fft.backward(out_buffer, result)

    except NotImplementedError as e:
        pytest.skip(str(e))

    local_diff = cp.max(cp.abs(ref_input - result)).item()
    global_diff = comm.allreduce(local_diff, op=MPI.MAX)

    tol = 1e-4 if "float32" in dtype_str else 1e-12
    assert global_diff < tol, f"Rank {comm.Get_rank()} failed with global max diff {global_diff:.2e}"
