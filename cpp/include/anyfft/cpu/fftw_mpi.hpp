/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "anyfft/core/fft_base.hpp"
#include "anyfft/core/mpi_utils.hpp"
#include <fftw3-mpi.h>

class FFTWMpiDist : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    FFTWMpiDist(
        const std::vector<int>& shape,
        py::array in,
        py::array out,
        const std::string& dtype,
        int comm_handle
    );

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    // Static helper: Asks FFTW how to slice the array for a given Rank
    // Result: tuple(local_in_shape, local_in_start, local_out_shape, local_out_start)
    static std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
    get_local_info(
        const std::vector<int>& shape,
        int comm_handle,
        bool r2c
    );

    ~FFTWMpiDist();
};
