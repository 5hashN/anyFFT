/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "fft_base.hpp"
#include "mpi_utils.hpp"
#include <cufftMp.h>

class CUFFT_MPI : public FFTBase {
    std::unique_ptr<FFTBase> impl_;

public:
    CUFFT_MPI(int ndim, const std::vector<int>& shape,
              int comm_handle, const std::string& dtype);

    ~CUFFT_MPI();

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    static std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
    get_local_info(int ndim, const std::vector<int>& global_shape, int comm_handle, bool r2c);
};