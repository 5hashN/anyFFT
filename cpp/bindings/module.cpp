/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#ifdef ENABLE_FFTW
#include "anyfft/cpu/fftw.hpp"
#endif

#ifdef ENABLE_FFTW_MPI
#include "anyfft/cpu/fftw_mpi.hpp"
#endif

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
#include "anyfft/gpu/gpufft.cuh"
#define HAS_GPUFFT 1
#else
#define HAS_GPUFFT 0
#endif

std::string get_gpu_backend_name() {
#if defined(ENABLE_CUDA)
    return "cuFFT";
#elif defined(ENABLE_HIP)
    return "hipFFT";
#else
    return "None";
#endif
}

#ifdef ENABLE_CUDA_MPI
#include "anyfft/gpu/cufftmp.cuh"
#endif


// Module Definition
PYBIND11_MODULE(_core, m) {

std::string gpu_name = get_gpu_backend_name();

std::string doc_str = "anyFFT Core Module\n"
                       "\n"
                       "High-performance C++ backend for serial and parallel FFT transforms.\n"
                       "This module exposes optimized bindings for CPU (FFTW) and GPU (cuFFT/hipFFT) libraries.\n\n"
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

#ifdef ENABLE_CUDA_MPI
    doc_str += "\n - [x] cuFFTMp (Distributed GPU)";
#else
    doc_str += "\n - [ ] cuFFTMp (Distributed GPU)";
#endif

#if HAS_GPUFFT
    doc_str += "\n - [x] " + gpu_name + " (Unified Serial GPU)";
#else
    doc_str += "\n - [ ] GPU FFT (Unified Serial GPU)";
#endif

    doc_str += "\n\nFor usage, please import the wrapper class:\n"
               "  from anyFFT import FFT";

    m.doc() = doc_str.c_str();

#ifdef ENABLE_FFTW
    m.attr("FFTW_ESTIMATE") = FFTW_ESTIMATE;
    m.attr("FFTW_MEASURE") = FFTW_MEASURE;
    m.attr("FFTW_PATIENT") = FFTW_PATIENT;
    m.attr("FFTW_EXHAUSTIVE") = FFTW_EXHAUSTIVE;

    // Guru Interface
    py::class_<FFTWLocal>(m, "fftw")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, py::array, py::array, const std::string&, int, unsigned>(),
            "shape"_a, "axes"_a, "input"_a, "output"_a, "dtype"_a, "n_threads"_a = 1, "flags"_a = FFTW_ESTIMATE,
            "FFTW backend. Supports:\n"
            " - Real-to-Complex (float64/32)\n"
            " - Complex-to-Complex (complex128/64)\n"
            " - In-Place & Out-of-Place\n"
            " - Arbitrary axes and strides")
        .def("forward", &FFTWLocal::forward)
        .def("backward", &FFTWLocal::backward);
#endif

#ifdef ENABLE_FFTW_MPI
    py::class_<FFTWMpiDist>(m, "fftw_mpi")
        .def(py::init<const std::vector<int>&, py::array, py::array, const std::string&, int>(),
            "global_shape"_a, "input"_a, "output"_a, "dtype"_a, "comm"_a,
            "FFTW-MPI Backend. Supports:\n"
            " - Real-to-Complex (In-Place ONLY)\n"
            " - Complex-to-Complex (In-Place & Out-of-Place)\n"
            " - 2D and 3D Slab Decomposition")
        .def("forward", &FFTWMpiDist::forward)
        .def("backward", &FFTWMpiDist::backward)
        .def_static("get_local_info", &FFTWMpiDist::get_local_info,
            "global_shape"_a, "comm"_a, "r2c"_a,
            "Returns (in_shape, in_start, out_shape, out_start). for slab decomposition.");
#endif

    m.def("get_gpu_backend_name", &get_gpu_backend_name, "Returns the compiled GPU backend name (cuFFT or hipFFT).");

#if HAS_GPUFFT
    py::class_<gpufftLocal>(m, "gpufft")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, const std::string&>(),
            "shape"_a, "axes"_a, "dtype"_a,
            (std::string("Unified GPU backend (") + gpu_name + "). Supports:\n"
            " - Real-to-Complex (float64/32)\n"
            " - Complex-to-Complex (complex128/64)\n"
            " - In-Place & Out-of-Place\n"
            " - Contiguous axes and strides").c_str())
        .def("forward", &gpufftLocal::forward)
        .def("backward", &gpufftLocal::backward);
#endif
}

#ifdef ENABLE_CUDA_MPI
    py::class_<cufftMpDist>(m, "cufftmp")
        .def(py::init<const std::vector<int>&, const std::vector<int>&, py::object, py::object, const std::string&, int>(),
            "global_shape"_a, "grid"_a, "input"_a, "output"_a, "dtype"_a, "comm"_a,
            "cuFFTMp Backend. Supports:\n"
            " - Real-to-Complex (In-Place & Out-of-Place)\n"
            " - Complex-to-Complex (In-Place & Out-of-Place)\n"
            " - Slab (1D) and Pencil (2D) Decomposition via 'grid' argument")
        .def("forward", &cufftMpDist::forward)
        .def("backward", &cufftMpDist::backward)
        .def_static("get_local_info", &cufftMpDist::get_local_info,
            "global_shape"_a, "grid"_a, "comm"_a, "r2c"_a,
            "Returns (in_shape, in_start, out_shape, out_start) for slab/pencil decomposition.");
#endif
