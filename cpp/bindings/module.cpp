#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef ENABLE_FFTW
#include "../fft/backends/fftw_serial.hpp"
#endif

#ifdef ENABLE_CUDA
#include "../fft/backends/cufft_serial.cuh"
#endif

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(anyFFT, m) {

#ifdef ENABLE_FFTW
    py::class_<FFTW_SERIAL>(m, "fftw")
        .def(py::init<int, const std::vector<int>&, py::array, py::array, const std::string&>(),
             "ndim"_a, "shape"_a, "real_in"_a, "complex_out"_a, "dtype"_a,
             "FFTW backend: Requires dummy arrays for plan creation.")
        .def("forward", &FFTW_SERIAL::forward)
        .def("backward", &FFTW_SERIAL::backward);
#endif

#ifdef ENABLE_CUDA
    py::class_<CUFFT_SERIAL>(m, "cufft")
        .def(py::init<int, const std::vector<int>&, const std::string&>(),
             "ndim"_a, "shape"_a, "dtype"_a,
             "cuFFT backend: Does NOT require dummy arrays.")
        .def("forward", &CUFFT_SERIAL::forward)
        .def("backward", &CUFFT_SERIAL::backward);
#endif

}