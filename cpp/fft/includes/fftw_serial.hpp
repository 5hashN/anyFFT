#pragma once
#include "fft_base.hpp"
#include <fftw3.h>
#include <cstring>

class FFTW_SERIAL : public FFTBase {
private:
    std::unique_ptr<FFTBase> impl_;

public:
    FFTW_SERIAL(int ndim,
                const std::vector<int>& shape,
                const std::vector<int>& axes,
                py::array dummy_real_in,
                py::array dummy_complex_out,
                const std::string& dtype);

    void forward(py::object real_in_obj, py::object complex_out_obj) override;
    void backward(py::object complex_in_obj, py::object real_out_obj) override;

    ~FFTW_SERIAL();
};
