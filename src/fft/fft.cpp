#include "fft.hpp"
#include <memory>
#include <stdexcept>

FFT::FFT(int ndim,
         const std::vector<int>& shape,
         py::array dummy_real_in,
         py::array dummy_complex_out,
         const std::string& dtype,
         const std::string& backend)
{
    if (backend == "fftw") {
        impl_ = std::make_unique<FFTW_SERIAL>(ndim, shape, dummy_real_in, dummy_complex_out, dtype);
    // } else if (backend == "fftw-mpi") {
    // } else if (backend == "cufft") {
    // } else if (backend == "cufftmp") {
    } else {
        throw std::runtime_error("Unknown FFT backend or not compiled");
    }
}

void FFT::forward(py::array real_in, py::array complex_out) {
    impl_->forward(real_in, complex_out);
}

void FFT::backward(py::array complex_in, py::array real_out) {
    impl_->backward(complex_in, real_out);
}
