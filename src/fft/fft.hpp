#pragma once
#include "fft_base.hpp"
#include "backends/fftw_serial.hpp"

class FFT {
private:
    std::unique_ptr<FFTBase> impl_;
public:
    FFT(int ndim,
        const std::vector<int>& shape,
        py::array dummy_real_in,
        py::array dummy_complex_out,
        const std::string& dtype,
        const std::string& backend);

    void forward(py::array real_in, py::array complex_out);
    void backward(py::array complex_in, py::array real_out);
};
