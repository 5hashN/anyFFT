#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <complex>
#include <string>
#include <tuple>

namespace py = pybind11;

class FFTBase {
public:
    virtual void forward(py::object real_in, py::object complex_out) = 0;
    virtual void backward(py::object complex_in, py::object real_out) = 0;
    virtual ~FFTBase() = default;
};