#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../fft/fft.hpp"

namespace py = pybind11;

PYBIND11_MODULE(anyFFT, m) {
    py::class_<FFT>(m, "FFT").def(py::init<
        int,
        const std::vector<int>&,
        py::array,
        py::array,
        const std::string&,
        const std::string&
        >(),
        py::arg("ndim"),
        py::arg("shape"),
        py::arg("input"),
        py::arg("output"),
        py::arg("dtype"),
        py::arg("backend")
    )
    .def("forward", &FFT::forward)
    .def("backward", &FFT::backward);
}
