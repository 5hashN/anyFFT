/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "anyfft/core/fft_base.hpp"
#include "anyfft/gpu/translations.cuh"

class gpufftLocal : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    gpufftLocal(
        const std::vector<int>& shape,
        const std::vector<int>& axes,
        const std::string& dtype
    );

    void forward(py::object in, py::object out) override;
    void backward(py::object in, py::object out) override;

    ~gpufftLocal();
};
