/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once
#include "common.hpp"

class FFTBase {
public:
    virtual void forward(py::object real_in, py::object complex_out) = 0;
    virtual void backward(py::object complex_in, py::object real_out) = 0;
    virtual ~FFTBase() = default;
};
