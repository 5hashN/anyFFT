/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "fft_base.hpp"
#include <fftw3.h>

class FFTW_SERIAL : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    FFTW_SERIAL(int ndim,
                const std::vector<int>& shape,
                const std::vector<int>& axes,
                py::array input,
                py::array output,
                const std::string& dtype,
                int n_threads,
                unsigned flags);

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    ~FFTW_SERIAL();
};
