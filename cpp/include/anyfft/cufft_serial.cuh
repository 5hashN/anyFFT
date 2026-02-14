/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "fft_base.hpp"
#include <cufft.h>
#include <cuda_runtime.h>

class CUFFT_SERIAL : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    CUFFT_SERIAL(const std::vector<int>& shape,
                 const std::vector<int>& axes,
                 const std::string& dtype);

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    ~CUFFT_SERIAL();
};
