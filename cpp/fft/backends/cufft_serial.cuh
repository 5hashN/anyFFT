#pragma once
#include "../includes/fft_base.hpp"
#include <cufft.h>
#include <cuda_runtime.h>

class CUFFT_SERIAL : public FFTBase {
private:
    std::vector<int> shape_;
    int ndim_;
    ssize_t N_;
    std::string dtype_;

    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;

    void forward_ptr(uintptr_t real_in_ptr, uintptr_t complex_out_ptr);
    void backward_ptr(uintptr_t complex_in_ptr, uintptr_t real_out_ptr);

public:
    CUFFT_SERIAL(int ndim,
                const std::vector<int>& shape,
                const std::string& dtype);

    void forward(py::object real_in, py::object complex_out) override;
    void backward(py::object complex_in, py::object real_out) override;

    ~CUFFT_SERIAL();
};