/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

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
#include <cstring>
#include <tuple>

#include <algorithm>
#include <numeric>
#include <set>
#include <functional>

namespace py = pybind11;

// Helper: Get element-wise strides from a numpy array
inline std::vector<int> get_array_strides(const py::array& arr) {
    std::vector<int> strides;
    int itemsize = arr.itemsize();
    for (ssize_t i = 0; i < arr.ndim(); ++i) {
        strides.push_back(arr.strides(i) / itemsize);
    }
    return strides;
}

// Helper: Recursive Scaler for Strided Arrays
inline void scale_recursive(char* ptr, int dim, int ndim,
                            const std::vector<ssize_t>& shape,
                            const std::vector<ssize_t>& strides,
                            double factor, int itemsize)
{
    if (dim == ndim - 1) {
        ssize_t n = shape[dim];
        ssize_t s = strides[dim];
        if (itemsize == 16) { // complex128
            for (ssize_t i = 0; i < n; ++i) {
                double* d = (double*)ptr;
                d[0] *= factor;
                d[1] *= factor;
                ptr += s;
            }
        } else if (itemsize == 8) { // float64 or complex64
            // Treat complex64 as 2 floats, or float64 as 1 double.
            // Scalar multiplication works the same way.
            if (s == 8) {
                for (ssize_t i=0; i<n; ++i) {
                    *(double*)ptr *= factor; ptr+=s;
                }
            } else {
                for (ssize_t i=0; i<n; ++i) {
                    float* f = (float*)ptr;
                    f[0] *= (float)factor;
                    f[1] *= (float)factor;
                    ptr+=s;
                }
            }
        } else if (itemsize == 4) { // float32
            for (ssize_t i = 0; i < n; ++i) {
                *(float*)ptr *= (float)factor;
                ptr += s;
            }
        }
        return;
    }
    for (ssize_t i = 0; i < shape[dim]; ++i) {
        scale_recursive(ptr, dim + 1, ndim, shape, strides, factor, itemsize);
        ptr += strides[dim];
    }
}

// Helper: Main Scaling Entry Point
inline void scale_array(py::array arr, double scale) {
    if (scale == 1.0) return;

    // Fast Path: Contiguous
    if (arr.flags() & py::array::c_style) {
        void* ptr = arr.mutable_data();
        ssize_t size = arr.size();
        if (arr.dtype().kind() == 'c') {
            size_t total = size * 2;
            if (arr.itemsize() == 16) {
                double* d = (double*)ptr;
                for(size_t i=0; i<total; ++i) {
                    d[i] *= scale;
                }
            } else {
                float* f = (float*)ptr;
                for(size_t i=0; i<total; ++i) {
                    f[i] *= (float)scale;
                }
            }
        } else {
            if (arr.itemsize() == 8) {
                double* d = (double*)ptr;
                for(size_t i=0; i<size; ++i) {
                    d[i] *= scale;
                }
            } else {
                float* f = (float*)ptr;
                for(size_t i=0; i<size; ++i) {
                    f[i] *= (float)scale;
                }
            }
        }
    } else {
        // Slow Path: Strided
        int ndim = arr.ndim();
        std::vector<ssize_t> shape(ndim), strides(ndim);
        for(int i=0; i<ndim; ++i) {
            shape[i]=arr.shape(i);
            strides[i]=arr.strides(i);
        }
        scale_recursive((char*)arr.mutable_data(), 0, ndim, shape, strides, scale, arr.itemsize());
    }
}

class FFTBase {
public:
    virtual void forward(py::object real_in, py::object complex_out) = 0;
    virtual void backward(py::object complex_in, py::object real_out) = 0;
    virtual ~FFTBase() = default;
};