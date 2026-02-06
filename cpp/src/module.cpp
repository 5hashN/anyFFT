/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef ENABLE_FFTW
#include "anyfft/fftw_serial.hpp"
#endif

#ifdef ENABLE_FFTW_MPI
#include "anyfft/fftw_mpi.hpp"
#endif

#ifdef ENABLE_CUDA
#include "anyfft/cufft_serial.cuh"
#endif

#ifdef ENABLE_CUDA_MPI
#include "anyfft/cufft_mpi.cuh"
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// Helper Subclasses
#ifdef ENABLE_FFTW
class FFTW_SERIAL_GENERIC : public FFTW_SERIAL {
public:
    using FFTW_SERIAL::FFTW_SERIAL;
};
#endif

#ifdef ENABLE_FFTW_MPI
class FFTW_MPI_GENERIC : public FFTW_MPI {
public:
    using FFTW_MPI::FFTW_MPI;
};
#endif

#ifdef ENABLE_CUDA
class CUFFT_SERIAL_GENERIC : public CUFFT_SERIAL {
public:
    using CUFFT_SERIAL::CUFFT_SERIAL;
};
#endif

// Module Definition
PYBIND11_MODULE(_core, m) {

std::string doc_str = "anyFFT Core Module\n"
                       "\n"
                       "High-performance C++ backend for serial and parallel FFT transforms.\n"
                       "This module exposes optimized bindings for CPU (FFTW) and GPU (cuFFT) libraries.\n\n"
                       "Compiled Configuration:\n"
                       "-----------------------";

#ifdef ENABLE_FFTW
    doc_str += "\n - [x] FFTW3 (Serial CPU)";
#else
    doc_str += "\n - [ ] FFTW3 (Serial CPU)";
#endif

#ifdef ENABLE_FFTW_MPI
    doc_str += "\n - [x] FFTW3-MPI (Distributed CPU)";
#else
    doc_str += "\n - [ ] FFTW3-MPI (Distributed CPU)";
#endif

#ifdef ENABLE_CUDA
    doc_str += "\n - [x] cuFFT (Serial GPU)";
#else
    doc_str += "\n - [ ] cuFFT (Serial GPU)";
#endif

#ifdef ENABLE_CUDA_MPI
    doc_str += "\n - [x] cuFFTMp (Distributed GPU)";
#else
    doc_str += "\n - [ ] cuFFTMp (Distributed GPU)";
#endif

    doc_str += "\n\nFor usage, please import the wrapper class:\n"
               "  from anyFFT import FFT";

    m.doc() = doc_str.c_str();

#ifdef ENABLE_FFTW
    m.attr("FFTW_ESTIMATE") = FFTW_ESTIMATE;
    m.attr("FFTW_MEASURE") = FFTW_MEASURE;
    m.attr("FFTW_PATIENT") = FFTW_PATIENT;
    m.attr("FFTW_EXHAUSTIVE") = FFTW_EXHAUSTIVE;

    py::class_<FFTW_SERIAL>(m, "fftw_serial")
        .def(py::init([](int ndim, const std::vector<int>& shape, py::array input, py::array output,
                         const std::string& dtype, int n_threads, unsigned flags) {
            return new FFTW_SERIAL(ndim, shape, {}, input, output, dtype, n_threads, flags);
        }),
            "ndim"_a, "shape"_a, "input"_a, "output"_a, "dtype"_a, "n_threads"_a = 1, "flags"_a = FFTW_ESTIMATE,
            "FFTW backend. Supports:\n"
            " - Real-to-Complex (float64/32)\n"
            " - Complex-to-Complex (complex128/64)\n"
            " - In-Place transforms (if input is output)")
        .def("forward", &FFTW_SERIAL::forward)
        .def("backward", &FFTW_SERIAL::backward);

    // Guru Interface
    py::class_<FFTW_SERIAL_GENERIC>(m, "fftw_serial_generic")
        .def(py::init([](const std::vector<int>& shape, const std::vector<int>& axes, py::array input, py::array output,
                         const std::string& dtype, int n_threads, unsigned flags) {
            return new FFTW_SERIAL_GENERIC(0, shape, axes, input, output, dtype, n_threads, flags);
        }),
            "shape"_a, "axes"_a, "input"_a, "output"_a, "dtype"_a, "n_threads"_a = 1, "flags"_a = FFTW_ESTIMATE,
            "Generic FFTW backend (Guru)")
        .def("forward", &FFTW_SERIAL_GENERIC::forward)
        .def("backward", &FFTW_SERIAL_GENERIC::backward);
#endif

#ifdef ENABLE_FFTW_MPI
    py::class_<FFTW_MPI>(m, "fftw_mpi")
        .def(py::init<int, const std::vector<int>&, py::array, py::array, int, const std::string&>(),
            "ndim"_a, "global_shape"_a, "input"_a, "output"_a, "comm"_a, "dtype"_a,
            "FFTW MPI Backend. Supports:\n"
            " - Real-to-Complex (In-Place ONLY)\n"
            " - Complex-to-Complex (In-Place & Out-of-Place)\n"
            " - 2D and 3D Slab Decomposition")
        .def("forward", &FFTW_MPI::forward)
        .def("backward", &FFTW_MPI::backward)
        .def_static("get_local_info", &FFTW_MPI::get_local_info,
            "ndim"_a, "global_shape"_a, "comm"_a, "r2c"_a,
            "Returns (in_shape, in_start, out_shape, out_start). for slab decomposition.");
#endif

#ifdef ENABLE_CUDA
    py::class_<CUFFT_SERIAL>(m, "cufft_serial")
        .def(py::init([](int ndim, const std::vector<int>& shape, const std::string& dtype) {
            return new CUFFT_SERIAL(ndim, shape, {}, dtype);
        }),
            "ndim"_a, "shape"_a, "dtype"_a,
            "cuFFT backend. Supports:\n"
            " - Real-to-Complex (float64/32)\n"
            " - Complex-to-Complex (complex128/64)\n"
            " - Plans are cached internally")
        .def("forward", &CUFFT_SERIAL::forward)
        .def("backward", &CUFFT_SERIAL::backward);

    py::class_<CUFFT_SERIAL_GENERIC>(m, "cufft_serial_generic")
        .def(py::init([](const std::vector<int>& shape, const std::vector<int>& axes, const std::string& dtype) {
            return new CUFFT_SERIAL_GENERIC(0, shape, axes, dtype);
        }),
            "shape"_a, "axes"_a, "dtype"_a,
            "Generic cuFFT backend (PlanMany)")
        .def("forward", &CUFFT_SERIAL_GENERIC::forward)
        .def("backward", &CUFFT_SERIAL_GENERIC::backward);
#endif

#ifdef ENABLE_CUDA_MPI
    py::class_<CUFFT_MPI>(m, "cufft_mpi")
        .def(py::init<int, const std::vector<int>&, int, const std::string&>(),
            "ndim"_a, "global_shape"_a, "comm"_a, "dtype"_a,
            "cuFFTMp Backend. Supports:\n"
            " - Real-to-Complex (In-Place ONLY)\n"
            " - Complex-to-Complex (In-Place & Out-of-Place)\n"
            " - 2D and 3D Slab Decomposition")
        .def("forward", &CUFFT_MPI::forward)
        .def("backward", &CUFFT_MPI::backward)
        .def_static("get_local_info", &CUFFT_MPI::get_local_info,
            "ndim"_a, "global_shape"_a, "comm"_a, "r2c"_a,
            "Returns (in_shape, in_start, out_shape, out_start) for slab decomposition.");
#endif
}