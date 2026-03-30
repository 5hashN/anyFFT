/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "anyfft/core/fft_base.hpp"
#include "anyfft/core/mpi_utils.hpp"
#include <cufftMp.h>

#ifdef USE_NVSHMEM
    #include <nvshmem.h>
    #include <nvshmemx.h>
#endif

class cufftMpDist : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    cufftMpDist(
        const std::vector<int>& shape,
        const std::vector<int>& grid,
        py::object in,
        py::object out,
        const std::string& dtype,
        int comm_handle
    );

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    static std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
    get_local_info(
        int ndim,
        const std::vector<int>& global_shape,
        const std::vector<int>& grid,
        int comm_handle,
        bool r2c
    );

    ~cufftMpDist();
};
