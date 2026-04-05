"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import pytest

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    pytest.skip("MPI library not functional on this system", allow_module_level=True)

import numpy as np
from .. import test_utils
import anyFFT
from anyFFT import FFT

if not anyFFT.has_backend("fftw_mpi"):
    pytest.skip("Skipping fftw_mpi tests: Backend not compiled", allow_module_level=True)

from anyFFT import fftw_mpi

BACKEND = "fftw_mpi"

@pytest.mark.mpi
@pytest.mark.cpu
@pytest.mark.parametrize("shape", [[1024, 1024], [128, 128, 128]])
@pytest.mark.parametrize("dtype_str", ["float64", "float32"])
@pytest.mark.parametrize("is_r2c", [False, True])
@pytest.mark.parametrize("is_inplace", [False, True])
def test_fftw_mpi_distributed(shape, dtype_str, is_r2c, is_inplace):
    comm = MPI.COMM_WORLD
    full_dtype_str = dtype_str if is_r2c else test_utils.get_c2c_dtype_str(dtype_str)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = fftw_mpi.get_local_info(shape, comm.py2f(), is_r2c)
    except NotImplementedError as e:
        pytest.skip(str(e))

    if not is_r2c:
        in_buffer = np.zeros(in_shape, dtype=complex_dtype, order="C")
        in_buffer[:] = test_utils.generate_data(in_shape, complex_dtype, start_indices=in_start)

        if is_inplace:
            out_buffer = in_buffer
            ref_input = in_buffer.copy()
        else:
            out_buffer = np.empty_like(in_buffer)
            ref_input = in_buffer.copy()

    else:
        if is_inplace:
            out_buffer = np.zeros(out_shape, dtype=complex_dtype, order="C")
            real_view = out_buffer.view(real_dtype)

            if len(shape) == 3:
                valid_slice = real_view[:, :, :shape[2]]
            else:
                valid_slice = real_view[:, :shape[1]]

            valid_slice[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=out_start)
            in_buffer = out_buffer
            ref_input = valid_slice.copy()
        else:
            in_buffer = np.zeros(in_shape, dtype=real_dtype, order="C")
            in_buffer[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=in_start)
            out_buffer = np.zeros(out_shape, dtype=complex_dtype, order="C")
            ref_input = in_buffer.copy()

    try:
        fft = FFT(shape=shape, input=in_buffer, output=out_buffer, comm=comm, dtype=full_dtype_str, backend=BACKEND)
        fft.forward(in_buffer, out_buffer)

        if is_inplace:
            fft.backward(out_buffer, in_buffer)
            result = valid_slice if (is_r2c and is_inplace) else in_buffer
        else:
            back_buffer = np.empty_like(in_buffer)
            fft.backward(out_buffer, back_buffer)
            result = back_buffer

    except NotImplementedError as e:
        pytest.skip(str(e))

    local_diff = np.max(np.abs(ref_input - result))
    global_diff = comm.allreduce(local_diff, op=MPI.MAX)

    tol = 1e-4 if "float32" in dtype_str else 1e-12
    assert global_diff < tol, f"Rank {comm.Get_rank()} failed with global max diff {global_diff:.2e}"
