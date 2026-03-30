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

TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = []


def run_fftw_mpi_test(global_shape, dtype_str, is_r2c=False, is_inplace=False):
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1

    test_type = "R2C" if is_r2c else "C2C"
    place_str = "In-Place" if is_inplace else "Out-Place"

    full_dtype_str = dtype_str if is_r2c else test_utils.get_c2c_dtype_str(dtype_str)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    test_id = f"[{test_type} {place_str}] {global_shape} | {full_dtype_str}"
    test_utils.print_test_start(f"[{test_type} {place_str}]", full_dtype_str, rank)

    try:
        in_shape, in_start, out_shape, out_start = anyFFT.fftw_mpi.get_local_info(
            global_shape, comm.py2f(), is_r2c
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    try:
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
                # R2C In-Place requires padded complex buffer
                out_buffer = np.zeros(out_shape, dtype=complex_dtype, order="C")
                real_view = out_buffer.view(real_dtype)

                # Slicing out the padding to write the real data
                if len(global_shape) == 3:
                    valid_slice = real_view[:, :, : global_shape[2]]
                else:
                    valid_slice = real_view[:, : global_shape[1]]

                valid_slice[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=out_start)
                in_buffer = out_buffer
                ref_input = valid_slice.copy()
            else:
                in_buffer = np.zeros(in_shape, dtype=real_dtype, order="C")
                in_buffer[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=in_start)
                out_buffer = np.zeros(out_shape, dtype=complex_dtype, order="C")
                ref_input = in_buffer.copy()

        fft = FFT(
            shape=global_shape, input=in_buffer, output=out_buffer,
            comm=comm, dtype=full_dtype_str, backend=BACKEND
        )

        fft.forward(in_buffer, out_buffer)

        if is_inplace:
            fft.backward(out_buffer, in_buffer)
            result = valid_slice if (is_r2c and is_inplace) else in_buffer
        else:
            back_buffer = np.empty_like(in_buffer)
            fft.backward(out_buffer, back_buffer)
            result = back_buffer

        local_diff = np.max(np.abs(ref_input - result))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)

        tol = 1e-4 if "float32" in dtype_str else 1e-12
        passed = test_utils.print_test_result(global_diff, tol=tol, rank=rank)

        if passed:
            PASSED_TESTS += 1
        else:
            FAILED_TESTS.append(f"{test_id} -> Failed (Diff: {global_diff:.2e})")

    except NotImplementedError as e:
        test_utils.print_skipped(str(e), rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)
        FAILED_TESTS.append(f"{test_id} -> Error: {str(e)}")


def main():
    test_utils.print_header(BACKEND, comm)
    configs = [(2, [1024, 1024]), (3, [128, 128, 128])]
    dtypes = ["float64", "float32"]

    for ndim, shape in configs:
        test_utils.print_section(f"Config: {ndim}D | Shape: {shape}", rank)
        for dtype in dtypes:
            run_fftw_mpi_test(shape, dtype, is_r2c=False, is_inplace=False)
            run_fftw_mpi_test(shape, dtype, is_r2c=False, is_inplace=True)
            run_fftw_mpi_test(shape, dtype, is_r2c=True,  is_inplace=False)
            run_fftw_mpi_test(shape, dtype, is_r2c=True,  is_inplace=True)

    test_utils.print_summary(TOTAL_TESTS, PASSED_TESTS, FAILED_TESTS, rank)

    if FAILED_TESTS and rank == 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
