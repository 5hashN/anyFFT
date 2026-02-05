/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "fft_base.hpp"
#include "mpi_utils.hpp"
#include <fftw3-mpi.h>

class FFTW_MPI : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    FFTW_MPI(int ndim,
             const std::vector<int>& global_shape,
             py::array input,
             py::array output,
             int comm_handle,
             const std::string& dtype);

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    // Static helper: Asks FFTW how to slice the array for a given Rank
    // Result: tuple(local_in_shape, local_in_start, local_out_shape, local_out_start)
    static std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
    get_local_info(int ndim, const std::vector<int>& global_shape, int comm_handle, bool r2c);

    ~FFTW_MPI();
};