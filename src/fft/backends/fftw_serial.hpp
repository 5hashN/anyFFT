#pragma once
#include "../fft_base.hpp"
#include <fftw3.h>
#include <string>
#include <vector>

class FFTW_SERIAL : public FFTBase {
private:
    std::vector<int> shape_;
    int ndim_;
    void* plan_forward_;
    void* plan_backward_;
    ssize_t N_;
    std::string dtype_;

public:
    FFTW_SERIAL(int ndim,
             const std::vector<int>& shape,
             py::array dummy_real_in,
             py::array dummy_complex_out,
             const std::string& dtype);

    void forward(py::array real_in, py::array complex_out) override;
    void backward(py::array complex_in, py::array real_out) override;

    ~FFTW_SERIAL();
};
