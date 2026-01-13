#pragma once
#include "../includes/fft_base.hpp"
#include <fftw3.h>

class FFTW_SERIAL : public FFTBase {
private:
    std::vector<int> shape_;
    int ndim_;
    ssize_t N_;
    std::string dtype_;

    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_SERIAL(int ndim,
             const std::vector<int>& shape,
             py::array dummy_real_in,
             py::array dummy_complex_out,
             const std::string& dtype);

    void forward(py::object real_in_obj, py::object complex_out_obj) override;
    void backward(py::object complex_in_obj, py::object real_out_obj) override;

    ~FFTW_SERIAL();
};
