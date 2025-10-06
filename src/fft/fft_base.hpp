#pragma once
#include <pybind11/numpy.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <complex>
#include <memory>
#include <vector>

namespace py = pybind11;

class FFTBase {
public:
    virtual void forward(py::array real_in, py::array complex_out) = 0;
    virtual void backward(py::array complex_in, py::array real_out) = 0;
    virtual ~FFTBase() = default;
};