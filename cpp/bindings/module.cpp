#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef ENABLE_FFTW
#include "../fft/includes/fftw_serial.hpp"
#endif

#ifdef ENABLE_FFTW_MPI
#include "../fft/includes/fftw_mpi.hpp"
#endif

#ifdef ENABLE_CUDA
#include "../fft/includes/cufft_serial.cuh"
#endif

#ifdef ENABLE_CUDA_MPI
#include "../fft/includes/cufft_mpi.cuh"
#endif

namespace py = pybind11;
using namespace pybind11::literals;

// Helper Subclasses
#ifdef ENABLE_FFTW
class FFTW_SERIAL_GURU : public FFTW_SERIAL {
public:
    using FFTW_SERIAL::FFTW_SERIAL;
};
#endif

#ifdef ENABLE_FFTW_MPI
class FFTW_MPI_GURU : public FFTW_MPI {
public:
    using FFTW_MPI::FFTW_MPI;
};
#endif

#ifdef ENABLE_CUDA
class CUFFT_SERIAL_AXES : public CUFFT_SERIAL {
public:
    using CUFFT_SERIAL::CUFFT_SERIAL;
};
#endif

// Module Definition
PYBIND11_MODULE(anyFFT, m) {

#ifdef ENABLE_FFTW
    py::class_<FFTW_SERIAL>(m, "fftw")
        .def(py::init([](int ndim, const std::vector<int>& shape, py::array input, py::array output, const std::string& dtype) {
            return new FFTW_SERIAL(ndim, shape, {}, input, output, dtype);
        }),
            "ndim"_a, "shape"_a, "input"_a, "output"_a, "dtype"_a,
            "FFTW backend. Supports:\n"
            " - Real-to-Complex (float64/32)\n"
            " - Complex-to-Complex (complex128/64)\n"
            " - In-Place transforms (if input is output)")
        .def("forward", &FFTW_SERIAL::forward)
        .def("backward", &FFTW_SERIAL::backward);

    // Guru Interface
    py::class_<FFTW_SERIAL_GURU>(m, "fftw_guru")
        .def(py::init([](const std::vector<int>& shape, const std::vector<int>& axes, py::array input, py::array output, const std::string& dtype) {
            return new FFTW_SERIAL_GURU(0, shape, axes, input, output, dtype);
        }),
            "shape"_a, "axes"_a, "input"_a, "output"_a, "dtype"_a,
            "Generic FFTW backend")
        .def("forward", &FFTW_SERIAL_GURU::forward)
        .def("backward", &FFTW_SERIAL_GURU::backward);
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
    py::class_<CUFFT_SERIAL>(m, "cufft")
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

    py::class_<CUFFT_SERIAL_AXES>(m, "cufft_generic")
        .def(py::init([](const std::vector<int>& shape, const std::vector<int>& axes, const std::string& dtype) {
            return new CUFFT_SERIAL_AXES(0, shape, axes, dtype);
        }),
            "shape"_a, "axes"_a, "dtype"_a,
            "Generic cuFFT backend (PlanMany)")
        .def("forward", &CUFFT_SERIAL_AXES::forward)
        .def("backward", &CUFFT_SERIAL_AXES::backward);
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