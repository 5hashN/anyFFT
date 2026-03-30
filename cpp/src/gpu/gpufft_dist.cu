/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/gpu/gpufft_dist.cuh"

template<typename T>
__global__ void pack_fwd_kernel(const T* in, T* out, int Nx_P, int Ny, int Ny_P, int Nz, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx_P * Ny * Nz) return;

    int z = idx % Nz;
    int tmp = idx / Nz;
    int y = tmp % Ny;
    int x_chunk = tmp / Ny;

    int p = y / Ny_P;
    int y_chunk = y % Ny_P;

    int out_idx = p * (Nx_P * Ny_P * Nz) + x_chunk * (Ny_P * Nz) + y_chunk * Nz + z;
    out[out_idx] = in[idx];
}

template<typename T>
__global__ void unpack_fwd_kernel(const T* in, T* out, int Nx, int Nx_P, int Ny_P, int Nz, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * Nx_P * Ny_P * Nz) return;

    int z = idx % Nz;
    int tmp = idx / Nz;
    int y_chunk = tmp % Ny_P;
    tmp /= Ny_P;
    int x_chunk = tmp % Nx_P;
    int p = tmp / Nx_P;

    int x = p * Nx_P + x_chunk;

    int out_idx = y_chunk * (Nz * Nx) + z * Nx + x;
    out[out_idx] = in[idx];
}

template<typename T>
__global__ void pack_bwd_kernel(const T* in, T* out, int Nx, int Nx_P, int Ny_P, int Nz, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ny_P * Nz * Nx) return;

    int x = idx % Nx;
    int tmp = idx / Nx;
    int z = tmp % Nz;
    int y_chunk = tmp / Nz;

    int p = x / Nx_P;
    int x_chunk = x % Nx_P;

    int out_idx = p * (Nx_P * Ny_P * Nz) + x_chunk * (Ny_P * Nz) + y_chunk * Nz + z;
    out[out_idx] = in[idx];
}

template<typename T>
__global__ void unpack_bwd_kernel(const T* in, T* out, int Nx_P, int Ny, int Ny_P, int Nz, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * Nx_P * Ny_P * Nz) return;

    int z = idx % Nz;
    int tmp = idx / Nz;
    int y_chunk = tmp % Ny_P;
    tmp /= Ny_P;
    int x_chunk = tmp % Nx_P;
    int p = tmp / Nx_P;

    int y = p * Ny_P + y_chunk;

    int out_idx = x_chunk * (Ny * Nz) + y * Nz + z;
    out[out_idx] = in[idx];
}

void* gpufftDist::get_gpu_ptr(py::object obj) {
    // Extract raw device pointer via Python's standard __cuda_array_interface__
    py::dict iface = obj.attr("__cuda_array_interface__");
    py::tuple data = iface["data"];
    return reinterpret_cast<void*>(data[0].cast<uintptr_t>());
}

gpufftDist::gpufftDist(
    const std::vector<int>& shape,
    const std::vector<int>& grid,
    py::object in,
    py::object out,
    const std::string& dtype,
    int comm_handle
)
{
    if (shape.size() != 3) {
        throw std::invalid_argument("1D Slab decomposition strictly requires a 3D global shape.");
    }

    comm = get_mpi_comm(comm_handle);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    Nx = shape[0];
    Ny = shape[1];
    Nz = shape[2];

    if (Nx % num_ranks != 0 || Ny % num_ranks != 0) {
        throw std::runtime_error("Nx and Ny must be evenly divisible by the number of MPI ranks for basic Slab decomposition.");
    }

    Nx_P = Nx / num_ranks;
    Ny_P = Ny / num_ranks;
    local_elements = Nx_P * Ny * Nz;

    is_double = (dtype == "complex128");
    element_size = is_double ? sizeof(gpufftDoubleComplex) : sizeof(gpufftComplex);
    transform_type_2d = is_double ? GPUFFT_Z2Z : GPUFFT_C2C;
    transform_type_1d = is_double ? GPUFFT_Z2Z : GPUFFT_C2C;

    int n2d[2] = {Ny, Nz};
    int inembed_2d[2] = {Ny, Nz};
    int idist_2d = Ny * Nz;
    int batch_2d = Nx_P;

    GPUFFT_CHECK(gpufftCreate(&plan_2d_YZ));
    GPUFFT_CHECK(gpufftPlanMany(&plan_2d_YZ, 2, n2d,
                                inembed_2d, 1, idist_2d,
                                inembed_2d, 1, idist_2d,
                                transform_type_2d, batch_2d));

    int n1d[1] = {Nx};
    int inembed_1d[1] = {Nx};
    int idist_1d = Nx;
    int batch_1d = Ny_P * Nz;

    GPUFFT_CHECK(gpufftCreate(&plan_1d_X));
    GPUFFT_CHECK(gpufftPlanMany(&plan_1d_X, 1, n1d,
                                inembed_1d, 1, idist_1d,
                                inembed_1d, 1, idist_1d,
                                transform_type_1d, batch_1d));

    allocate_network_buffers();
}

gpufftDist::~gpufftDist() {
    gpufftDestroy(plan_2d_YZ);
    gpufftDestroy(plan_1d_X);
    free_network_buffers();
}

void gpufftDist::allocate_network_buffers() {
    GPUFFT_CHECK(gpufftMalloc(&d_send_buffer, local_elements * element_size));
    GPUFFT_CHECK(gpufftMalloc(&d_recv_buffer, local_elements * element_size));
}

void gpufftDist::free_network_buffers() {
    GPUFFT_CHECK(gpufftFree(d_send_buffer));
    GPUFFT_CHECK(gpufftFree(d_recv_buffer));
}

void gpufftDist::forward(py::object in, py::object out) {
    void* d_input = get_gpu_ptr(in);
    void* d_output = get_gpu_ptr(out);

    int threads = 256;
    int blocks = (local_elements + threads - 1) / threads;
    int count_per_rank = (Nx_P * Ny_P * Nz) * element_size;

    if (is_double) {
        GPUFFT_CHECK(gpufftExecZ2Z(plan_2d_YZ, (gpufftDoubleComplex*)d_input, (gpufftDoubleComplex*)d_input, GPUFFT_FORWARD));
    } else {
        GPUFFT_CHECK(gpufftExecC2C(plan_2d_YZ, (gpufftComplex*)d_input, (gpufftComplex*)d_input, GPUFFT_FORWARD));
    }
    gpufftDeviceSynchronize();

    if (is_double) {
        pack_fwd_kernel<<<blocks, threads>>>((gpufftDoubleComplex*)d_input, (gpufftDoubleComplex*)d_send_buffer, Nx_P, Ny, Ny_P, Nz, num_ranks);
    } else {
        pack_fwd_kernel<<<blocks, threads>>>((gpufftComplex*)d_input, (gpufftComplex*)d_send_buffer, Nx_P, Ny, Ny_P, Nz, num_ranks);
    }
    gpufftDeviceSynchronize();

    MPI_Alltoall(d_send_buffer, count_per_rank, MPI_BYTE,
                 d_recv_buffer, count_per_rank, MPI_BYTE, comm);

    if (is_double) {
        unpack_fwd_kernel<<<blocks, threads>>>((gpufftDoubleComplex*)d_recv_buffer, (gpufftDoubleComplex*)d_output, Nx, Nx_P, Ny_P, Nz, num_ranks);
    } else {
        unpack_fwd_kernel<<<blocks, threads>>>((gpufftComplex*)d_recv_buffer, (gpufftComplex*)d_output, Nx, Nx_P, Ny_P, Nz, num_ranks);
    }
    gpufftDeviceSynchronize();

    if (is_double) {
        GPUFFT_CHECK(gpufftExecZ2Z(plan_1d_X, (gpufftDoubleComplex*)d_output, (gpufftDoubleComplex*)d_output, GPUFFT_FORWARD));
    } else {
        GPUFFT_CHECK(gpufftExecC2C(plan_1d_X, (gpufftComplex*)d_output, (gpufftComplex*)d_output, GPUFFT_FORWARD));
    }
}

void gpufftDist::backward(py::object in, py::object out) {
    void* d_input = get_gpu_ptr(in);
    void* d_output = get_gpu_ptr(out);

    int threads = 256;
    int blocks = (local_elements + threads - 1) / threads;
    int count_per_rank = (Nx_P * Ny_P * Nz) * element_size;

    if (is_double) {
        GPUFFT_CHECK(gpufftExecZ2Z(plan_1d_X, (gpufftDoubleComplex*)d_input, (gpufftDoubleComplex*)d_input, GPUFFT_INVERSE));
    } else {
        GPUFFT_CHECK(gpufftExecC2C(plan_1d_X, (gpufftComplex*)d_input, (gpufftComplex*)d_input, GPUFFT_INVERSE));
    }
    gpufftDeviceSynchronize();

    if (is_double) {
        pack_bwd_kernel<<<blocks, threads>>>((gpufftDoubleComplex*)d_input, (gpufftDoubleComplex*)d_send_buffer, Nx, Nx_P, Ny_P, Nz, num_ranks);
    } else {
        pack_bwd_kernel<<<blocks, threads>>>((gpufftComplex*)d_input, (gpufftComplex*)d_send_buffer, Nx, Nx_P, Ny_P, Nz, num_ranks);
    }
    gpufftDeviceSynchronize();

    MPI_Alltoall(d_send_buffer, count_per_rank, MPI_BYTE,
                 d_recv_buffer, count_per_rank, MPI_BYTE, comm);

    if (is_double) {
        unpack_bwd_kernel<<<blocks, threads>>>((gpufftDoubleComplex*)d_recv_buffer, (gpufftDoubleComplex*)d_output, Nx_P, Ny, Ny_P, Nz, num_ranks);
    } else {
        unpack_bwd_kernel<<<blocks, threads>>>((gpufftComplex*)d_recv_buffer, (gpufftComplex*)d_output, Nx_P, Ny, Ny_P, Nz, num_ranks);
    }
    gpufftDeviceSynchronize();

    if (is_double) {
        GPUFFT_CHECK(gpufftExecZ2Z(plan_2d_YZ, (gpufftDoubleComplex*)d_output, (gpufftDoubleComplex*)d_output, GPUFFT_INVERSE));
    } else {
        GPUFFT_CHECK(gpufftExecC2C(plan_2d_YZ, (gpufftComplex*)d_output, (gpufftComplex*)d_output, GPUFFT_INVERSE));
    }
}
