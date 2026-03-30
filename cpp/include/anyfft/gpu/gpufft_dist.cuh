/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once

#include "anyfft/core/fft_base.hpp"
#include "anyfft/core/mpi_utils.hpp"
#include "anyfft/gpu/translations.cuh"

class gpufftDist : public FFTBase {
private:
    gpufftHandle plan_2d_YZ;
    gpufftHandle plan_1d_X;

    MPI_Comm comm;
    int rank;
    int num_ranks;

    int Nx, Ny, Nz;

    int Nx_P, Ny_P;

    int local_elements;

    gpufftType transform_type_2d;
    gpufftType transform_type_1d;
    bool is_double;
    size_t element_size;

    void* d_send_buffer;
    void* d_recv_buffer;

    void allocate_network_buffers();
    void free_network_buffers();

    void* get_gpu_ptr(py::object obj);

public:
    gpufftDist(
        const std::vector<int>& shape,
        const std::vector<int>& grid,
        py::object in,
        py::object out,
        const std::string& dtype,
        int comm_handle
    );

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    ~gpufftDist();
};
