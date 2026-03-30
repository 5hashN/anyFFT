/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "anyfft/core/fft_base.hpp"
#include <fftw3.h>

class FFTWLocal : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    FFTWLocal(
        const std::vector<int>& shape,
        const std::vector<int>& axes,
        py::array in,
        py::array out,
        const std::string& dtype,
        int n_threads,
        unsigned flags
    );

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    ~FFTWLocal();
};
