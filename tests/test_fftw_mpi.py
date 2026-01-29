from mpi4py import MPI
import numpy as np
import sys
import test_utils

try:
    import anyFFT
    from anyFFT import FFT
except ImportError:
    print("Error: Could not import 'anyFFT'.")
    sys.exit(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
BACKEND = "fftw_mpi"

def test_r2c_out_of_place(dtype_str, global_shape, ndim):
    test_utils.print_test_start("[R2C Out-Place]", dtype_str, rank)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    try:
        in_shape, in_start, out_shape, out_start = anyFFT.fftw_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    # Alloc
    local_real = np.zeros(in_shape, dtype=real_dtype, order='C')
    local_complex = np.zeros(out_shape, dtype=complex_dtype, order='C')
    local_back = np.empty_like(local_real)

    # Fill
    local_real[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=in_start)
    local_real_ref = local_real.copy()

    # Exec
    try:
        fft = FFT(ndim, global_shape, input=local_real, output=local_complex,
                  comm=comm, dtype=dtype_str, backend=BACKEND)
        fft.forward(local_real, local_complex)
        fft.backward(local_complex, local_back)

        diff = np.max(np.abs(local_real_ref - local_back))
        global_diff = comm.allreduce(diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)

def test_r2c_in_place(dtype_str, global_shape, ndim):
    test_utils.print_test_start("[R2C In-Place ]", dtype_str, rank)
    real_dtype, complex_dtype = test_utils.get_numpy_types(dtype_str)

    try:
        # Get layout. 'out_shape' is the padded complex size.
        in_shape, in_start, out_shape, out_start = anyFFT.fftw_mpi.get_local_info(
            ndim, global_shape, comm.py2f(), True
        )
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    data_buffer = np.zeros(out_shape, dtype=complex_dtype, order='C')
    real_view = data_buffer.view(real_dtype)

    # Identify valid slice in padded buffer
    if ndim == 3:
        valid_input_slice = real_view[:, :, :global_shape[2]]
    else:
        valid_input_slice = real_view[:, :global_shape[1]]

    # Fill (using 'out_start' because standard slab input start == output start)
    valid_input_slice[:] = test_utils.generate_data(in_shape, real_dtype, start_indices=out_start)
    clean_input = valid_input_slice.copy()

    try:
        fft = FFT(ndim, global_shape, input=data_buffer, output=data_buffer,
                  comm=comm, dtype=dtype_str, backend=BACKEND)
        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = np.max(np.abs(clean_input - valid_input_slice))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)

def test_c2c_out_of_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C Out-Place]", c_dtype_str, rank)

    _, complex_dtype = test_utils.get_numpy_types(real_dtype_str)

    try:
        in_shape, in_start, _, _ = anyFFT.fftw_mpi.get_local_info(ndim, global_shape, comm.py2f(), False)
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    local_in = np.zeros(in_shape, dtype=complex_dtype, order='C')
    local_out = np.empty_like(local_in)
    local_back = np.empty_like(local_in)

    local_in[:] = test_utils.generate_data(in_shape, complex_dtype, start_indices=in_start)
    local_in_ref = local_in.copy()

    try:
        fft = FFT(ndim, global_shape, input=local_in, output=local_out,
                  comm=comm, dtype=c_dtype_str, backend=BACKEND)
        fft.forward(local_in, local_out)
        fft.backward(local_out, local_back)

        local_diff = np.max(np.abs(local_in_ref - local_back))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)

def test_c2c_in_place(real_dtype_str, global_shape, ndim):
    c_dtype_str = test_utils.get_c2c_dtype_str(real_dtype_str)
    test_utils.print_test_start("[C2C In-Place ]", c_dtype_str, rank)

    _, complex_dtype = test_utils.get_numpy_types(real_dtype_str)

    try:
        in_shape, in_start, _, _ = anyFFT.fftw_mpi.get_local_info(ndim, global_shape, comm.py2f(), False)
    except Exception as e:
        test_utils.print_skipped(str(e), rank)
        return

    data_buffer = np.zeros(in_shape, dtype=complex_dtype, order='C')
    data_buffer[:] = test_utils.generate_data(in_shape, complex_dtype, start_indices=in_start)
    clean_input = data_buffer.copy()

    try:
        fft = FFT(ndim, global_shape, input=data_buffer, output=data_buffer,
                  comm=comm, dtype=c_dtype_str, backend=BACKEND)
        fft.forward(data_buffer, data_buffer)
        fft.backward(data_buffer, data_buffer)

        local_diff = np.max(np.abs(clean_input - data_buffer))
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        test_utils.print_test_result(global_diff, rank=rank)
    except Exception as e:
        test_utils.print_test_result(0, rank=rank, error=e)

def main():
    test_utils.print_header(BACKEND, comm)
    precisions = ["float64", "float32"]
    configurations = [(2, [1024, 1024]), (3, [128, 128, 128])]

    for ndim, shape in configurations:
        test_utils.print_config(ndim, shape, rank)
        for dtype in precisions:
            if rank == 0: print(f"Precision: {dtype}")
            test_r2c_out_of_place(dtype, shape, ndim)
            test_r2c_in_place(dtype, shape, ndim)
            test_c2c_out_of_place(dtype, shape, ndim)
            test_c2c_in_place(dtype, shape, ndim)
            if rank == 0: print("")

    if rank == 0:
        print("Tests Completed.")

if __name__ == "__main__":
    main()