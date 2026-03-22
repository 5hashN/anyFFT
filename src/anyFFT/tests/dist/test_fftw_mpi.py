"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

from mpi4py import MPI
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_utils

try:
    import anyFFT
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
BACKEND = "fftw_mpi"


def test_r2c_out_of_place(dtype_str, global_shape, ndim):
    test_name = "[R2C Out-Place]"
    test_id = f"{test_name} {global_shape} {dtype_str}"
    test_utils.print_test_start(test_name, dtype_str, rank)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = anyFFT.fftw_mpi.get_local_info(
            global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return None

    local_real = np.zeros(in_shape, dtype=real_dtype, order="C")
    local_complex = np.zeros(out_shape, dtype=complex_dtype, order="C")
    local_back = np.empty_like(local_real)

    local_real[:] = test_utils.generate_data(
        in_shape, real_dtype, start_indices=in_start
    )
    local_real_ref = local_real.copy()

    try:
        fft = FFT(
            shape=global_shape, input=local_real, output=local_complex,
            comm=comm, dtype=dtype_str, backend=BACKEND
        )
        fft.forward(local_real, local_complex)
        fft.backward(local_complex, local_back)

        diff = np.max(np.abs(local_real_ref - local_back))
        global_diff = comm.allreduce(diff, op=MPI.MAX)
        passed = test_utils.print_test_result(global_diff, rank=rank)

        return (passed, test_id, f"Diff: {global_diff:.2e}")

    except NotImplementedError as e:
        test_utils.print_skipped(str(e), rank)
        return None

    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)
        return (False, test_id, str(e))


def test_r2c_in_place(dtype_str, global_shape, ndim):
    test_name = "[R2C In-Place ]"
    test_id = f"{test_name} {global_shape} {dtype_str}"
    test_utils.print_test_start(test_name, dtype_str, rank)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = anyFFT.fftw_mpi.get_local_info(
            global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return None

    data_buffer = np.zeros(out_shape, dtype=complex_dtype, order="C")
    real_view = data_buffer.view(real_dtype)

    if ndim == 3:
        valid_input_slice = real_view[:, :, : global_shape[2]]
    else:
        valid_input_slice = real_view[:, : global_shape[1]]

    valid_input_slice[:] = test_utils.generate_data(
        in_shape, real_dtype, start_indices=out_start
    )
    clean_input = valid_input_slice.copy()

    try:
        fft = FFT(
            shape=global_shape, input=data_buffer, output=data_buffer,
            comm=comm, dtype=dtype_str, backend=BACKEND
        )
        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = np.max(np.abs(clean_input - valid_input_slice))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        passed = test_utils.print_test_result(global_diff, rank=rank)

        return (passed, test_id, f"Diff: {global_diff:.2e}")

    except NotImplementedError as e:
        test_utils.print_skipped(str(e), rank)
        return None

    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)
        return (False, test_id, str(e))


def test_c2c_out_of_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_name = "[C2C Out-Place]"
    test_id = f"{test_name} {global_shape} {c_dtype_str}"
    test_utils.print_test_start(test_name, c_dtype_str, rank)

    _, complex_dtype = test_utils.get_numpy_types(real_dtype_str)

    try:
        in_shape, in_start, _, _ = anyFFT.fftw_mpi.get_local_info(
            global_shape, comm.py2f(), False
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return None

    local_in = np.zeros(in_shape, dtype=complex_dtype, order="C")
    local_out = np.empty_like(local_in)
    local_back = np.empty_like(local_in)

    local_in[:] = test_utils.generate_data(
        in_shape, complex_dtype, start_indices=in_start
    )
    local_in_ref = local_in.copy()

    try:
        fft = FFT(
            shape=global_shape, input=local_in, output=local_out,
            comm=comm, dtype=c_dtype_str, backend=BACKEND
        )
        fft.forward(local_in, local_out)
        fft.backward(local_out, local_back)

        local_diff = np.max(np.abs(local_in_ref - local_back))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        passed = test_utils.print_test_result(global_diff, rank=rank)

        return (passed, test_id, f"Diff: {global_diff:.2e}")

    except NotImplementedError as e:
        test_utils.print_skipped(str(e), rank)
        return None

    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)
        return (False, test_id, str(e))


def test_c2c_in_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_name = "[C2C In-Place ]"
    test_id = f"{test_name} {global_shape} {c_dtype_str}"
    test_utils.print_test_start(test_name, c_dtype_str, rank)

    _, complex_dtype = test_utils.get_numpy_types(real_dtype_str)

    try:
        in_shape, in_start, _, _ = anyFFT.fftw_mpi.get_local_info(
            global_shape, comm.py2f(), False
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return None

    data_buffer = np.zeros(in_shape, dtype=complex_dtype, order="C")
    data_buffer[:] = test_utils.generate_data(
        in_shape, complex_dtype, start_indices=in_start
    )
    clean_input = data_buffer.copy()

    try:
        fft = FFT(
            shape=global_shape, input=data_buffer, output=data_buffer,
            comm=comm, dtype=c_dtype_str, backend=BACKEND
        )
        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = np.max(np.abs(clean_input - data_buffer))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        passed = test_utils.print_test_result(global_diff, rank=rank)

        return (passed, test_id, f"Diff: {global_diff:.2e}")

    except NotImplementedError as e:
        test_utils.print_skipped(str(e), rank)
        return None

    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)
        return (False, test_id, str(e))


def main():
    test_utils.print_header(BACKEND, comm)
    configs = [(2, [1024, 1024]), (3, [128, 128, 128])]
    dtypes = ["float64", "float32"]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    def run_t(func, *args):
        nonlocal total_tests, passed_tests, failed_tests
        res = func(*args)
        if res is None: return
        total_tests += 1
        is_pass, t_id, err_msg = res
        if is_pass:
            passed_tests += 1
        else:
            failed_tests.append(f"{t_id} -> {err_msg}")

    for ndim, shape in configs:
        test_utils.print_section(f"Config: {ndim}D | Shape: {shape}", rank)
        for dtype in dtypes:
            run_t(test_r2c_out_of_place, dtype, shape, ndim)
            run_t(test_r2c_in_place, dtype, shape, ndim)
            run_t(test_c2c_out_of_place, dtype, shape, ndim)
            run_t(test_c2c_in_place, dtype, shape, ndim)

    test_utils.print_summary(total_tests, passed_tests, failed_tests, rank)

    if failed_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
