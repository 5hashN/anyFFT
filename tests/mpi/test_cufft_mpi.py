from mpi4py import MPI
import numpy as np
import cupy as cp
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
BACKEND = "cufft_mpi"


def test_c2c_in_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C In-Place ]", c_dtype_str, rank)
    _, complex_dtype = test_utils.get_cupy_types(real_dtype_str)

    # Get Layout (returns identical shape for C2C)
    try:
        in_shape, in_start, _, _ = anyFFT.cufft_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), False
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    # Alloc
    data_buffer = cp.zeros(in_shape, dtype=complex_dtype)

    # Fill
    host_data = test_utils.generate_data(
        in_shape,
        np.complex128 if real_dtype_str == "float64" else np.complex64,
        start_indices=in_start,
    )
    data_buffer[:] = cp.asarray(host_data)
    clean_input = data_buffer.copy()

    # Exec
    try:
        fft = FFT(
            ndim,
            global_shape,
            input=data_buffer,
            output=data_buffer,
            comm=comm,
            dtype=c_dtype_str,
            backend=BACKEND,
        )

        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = cp.max(cp.abs(clean_input - data_buffer)).item()
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)


def test_c2c_out_of_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C Out-Place]", c_dtype_str, rank)
    _, complex_dtype = test_utils.get_cupy_types(real_dtype_str)

    try:
        in_shape, in_start, _, _ = anyFFT.cufft_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), False
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    local_in = cp.zeros(in_shape, dtype=complex_dtype)
    local_out = cp.empty_like(local_in)
    local_back = cp.empty_like(local_in)

    host_data = test_utils.generate_data(
        in_shape,
        np.complex128 if real_dtype_str == "float64" else np.complex64,
        start_indices=in_start,
    )
    local_in[:] = cp.asarray(host_data)
    local_in_ref = local_in.copy()

    try:
        fft = FFT(
            ndim,
            global_shape,
            input=local_in,
            output=local_out,
            comm=comm,
            dtype=c_dtype_str,
            backend=BACKEND,
        )

        fft.forward(local_in, local_out)
        fft.backward(local_out, local_back)

        local_diff = cp.max(cp.abs(local_in_ref - local_back)).item()
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)


def test_r2c_in_place(dtype_str, global_shape, ndim):
    test_utils.print_test_start("[R2C In-Place ]", dtype_str, rank)

    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    # Numpy types for generation
    np_real, _ = test_utils.get_numpy_types(dtype_str)

    # Get Layout (Returns PADDED Complex size)
    try:
        in_shape, in_start, out_shape, out_start = anyFFT.cufft_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    # Alloc
    data_buffer = cp.zeros(out_shape, dtype=complex_dtype)
    real_view = data_buffer.view(real_dtype)

    # View Slicing (Valid Real portion)
    if ndim == 3:
        valid_slice = real_view[:, :, : global_shape[2]]
    else:
        valid_slice = real_view[:, : global_shape[1]]

    # Fill
    host_data = test_utils.generate_data(in_shape, np_real, start_indices=out_start)
    valid_slice[:] = cp.asarray(host_data)
    clean_input = valid_slice.copy()

    # Exec
    try:
        fft = FFT(
            ndim,
            global_shape,
            input=data_buffer,
            output=data_buffer,
            comm=comm,
            dtype=dtype_str,
            backend=BACKEND,
        )

        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = cp.max(cp.abs(clean_input - valid_slice)).item()
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)


def test_r2c_out_of_place(dtype_str, global_shape, ndim):
    test_utils.print_test_start("[R2C Out-Place]", dtype_str, rank)
    real_dtype, complex_dtype = test_utils.get_cupy_types(dtype_str)
    np_real, _ = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = anyFFT.cufft_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    # Alloc
    local_real = cp.zeros(in_shape, dtype=real_dtype)
    local_complex = cp.zeros(out_shape, dtype=complex_dtype)
    local_back = cp.empty_like(local_real)

    # Fill
    host_data = test_utils.generate_data(in_shape, np_real, start_indices=in_start)
    local_real[:] = cp.asarray(host_data)
    local_real_ref = local_real.copy()

    # Exec
    try:
        fft = FFT(
            ndim,
            global_shape,
            input=local_real,
            output=local_complex,
            comm=comm,
            dtype=dtype_str,
            backend=BACKEND,
        )
        fft.forward(local_real, local_complex)
        fft.backward(local_complex, local_back)

        local_diff = cp.max(cp.abs(local_real_ref - local_back)).item()
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)


def main():
    test_utils.print_header(BACKEND, comm)

    configs = [(2, [1024, 1024]), (3, [128, 128, 128])]
    dtypes = ["float64", "float32"]

    for ndim, shape in configs:
        test_utils.print_config(ndim, shape, rank)
        for dtype in dtypes:
            if rank == 0:
                print(f"Precision: {dtype}")
            test_r2c_out_of_place(dtype, shape, ndim)
            test_r2c_in_place(dtype, shape, ndim)
            test_c2c_out_of_place(dtype, shape, ndim)
            test_c2c_in_place(dtype, shape, ndim)
            if rank == 0:
                print("")

    if rank == 0:
        print("Tests Completed.")


if __name__ == "__main__":
    main()
