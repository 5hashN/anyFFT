#include "fftw_serial.hpp"
#include <stdexcept>

FFTW_SERIAL::FFTW_SERIAL(int ndim,
            const std::vector<int>& shape,
            py::array dummy_real_in,
            py::array dummy_complex_out,
            const std::string& dtype)
    : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
{
    // Validate array shapes/dtypes
    if (dummy_real_in.ndim() != ndim_)
        throw std::runtime_error("Dummy real array ndim mismatch");
    if (dummy_complex_out.ndim() != ndim_)
        throw std::runtime_error("Dummy complex array ndim mismatch");

    for (int i = 0; i < ndim_; ++i) {
        if (dummy_real_in.shape(i) != shape_[i])
            throw std::runtime_error("Dummy real array shape does not match dimensions");
    }

    // For r2c, last dimension is n//2 + 1
    if (dummy_complex_out.shape(0) != shape_[0])
        throw std::runtime_error("Dummy complex array first dim mismatch");
    if (dummy_complex_out.shape(ndim_-1) != shape_[ndim_-1]/2 + 1)
        throw std::runtime_error("Dummy complex array last dim should be n//2+1");

    for (int i = 0; i < ndim_; ++i) N_ *= shape_[i];

    // Precision selection
    if (dtype == "float64") {
        if (dummy_real_in.dtype().kind() != 'f' || dummy_real_in.dtype().itemsize() != 8)
            throw std::runtime_error("Dummy real array must be float64");
        if (dummy_complex_out.dtype().kind() != 'c' || dummy_complex_out.dtype().itemsize() != 16)
            throw std::runtime_error("Dummy complex array must be complex128");

        // Create plan with correct dimensionality
        if (ndim_ == 1) {
            plan_r2c_ = fftw_plan_dft_r2c_1d(
                shape_[0],
                static_cast<double*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftw_plan_dft_c2r_1d(
                shape_[0],
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                static_cast<double*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else if (ndim_ == 2) {
            plan_r2c_ = fftw_plan_dft_r2c_2d(
                shape_[0], shape_[1],
                static_cast<double*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftw_plan_dft_c2r_2d(
                shape_[0], shape_[1],
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                static_cast<double*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else if (ndim_ == 3) {
            plan_r2c_ = fftw_plan_dft_r2c_3d(
                shape_[0], shape_[1], shape_[2],
                static_cast<double*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftw_plan_dft_c2r_3d(
                shape_[0], shape_[1], shape_[2],
                reinterpret_cast<fftw_complex*>(dummy_complex_out.mutable_data()),
                static_cast<double*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else {
            throw std::runtime_error("ND not implemented for double precision");
        }
    } else if (dtype == "float32") {
        if (dummy_real_in.dtype().kind() != 'f' || dummy_real_in.dtype().itemsize() != 4)
            throw std::runtime_error("Dummy real array must be float32");
        if (dummy_complex_out.dtype().kind() != 'c' || dummy_complex_out.dtype().itemsize() != 8)
            throw std::runtime_error("Dummy complex array must be complex64");

        // Create plan with correct dimensionality
        if (ndim_ == 1) {
            plan_r2c_ = fftwf_plan_dft_r2c_1d(
                shape_[0],
                static_cast<float*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftwf_plan_dft_c2r_1d(
                shape_[0],
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                static_cast<float*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else if (ndim_ == 2) {
            plan_r2c_ = fftwf_plan_dft_r2c_2d(
                shape_[0], shape_[1],
                static_cast<float*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftwf_plan_dft_c2r_2d(
                shape_[0], shape_[1],
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                static_cast<float*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else if (ndim_ == 3) {
            plan_r2c_ = fftwf_plan_dft_r2c_3d(
                shape_[0], shape_[1], shape_[2],
                static_cast<float*>(dummy_real_in.mutable_data()),
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                FFTW_ESTIMATE
            );
            plan_c2r_ = fftwf_plan_dft_c2r_3d(
                shape_[0], shape_[1], shape_[2],
                reinterpret_cast<fftwf_complex*>(dummy_complex_out.mutable_data()),
                static_cast<float*>(dummy_real_in.mutable_data()),
                FFTW_ESTIMATE
            );
        } else {
            throw std::runtime_error("ND not implemented for single precision");
        }
    } else {
        throw std::runtime_error("Unknown or unsupported datatype");
    }
}

void FFTW_SERIAL::forward(py::object real_in_obj, py::object complex_out_obj) {
    py::array real_in = real_in_obj.cast<py::array>();
    py::array complex_out = complex_out_obj.cast<py::array>();

    if (dtype_ == "float64") {
        fftw_execute_dft_r2c(
            static_cast<fftw_plan>(plan_r2c_),
            static_cast<double*>(real_in.mutable_data()),
            reinterpret_cast<fftw_complex*>(complex_out.mutable_data())
        );
    } else if (dtype_ == "float32") {
        fftwf_execute_dft_r2c(
            static_cast<fftwf_plan>(plan_r2c_),
            static_cast<float*>(real_in.mutable_data()),
            reinterpret_cast<fftwf_complex*>(complex_out.mutable_data())
        );
    }
}

void FFTW_SERIAL::backward(py::object complex_in_obj, py::object real_out_obj) {
    py::array complex_in = complex_in_obj.cast<py::array>();
    py::array real_out = real_out_obj.cast<py::array>();

    if (dtype_ == "float64") {
        fftw_execute_dft_c2r(
            static_cast<fftw_plan>(plan_c2r_),
            reinterpret_cast<fftw_complex*>(complex_in.mutable_data()),
            static_cast<double*>(real_out.mutable_data())
        );
        double* buf = static_cast<double*>(real_out.mutable_data());
        for (ssize_t i = 0; i < real_out.size(); ++i) buf[i] /= N_;
    } else if (dtype_ == "float32") {
        fftwf_execute_dft_c2r(
            static_cast<fftwf_plan>(plan_c2r_),
            reinterpret_cast<fftwf_complex*>(complex_in.mutable_data()),
            static_cast<float*>(real_out.mutable_data())
        );
        float* buf = static_cast<float*>(real_out.mutable_data());
        for (ssize_t i = 0; i < real_out.size(); ++i) buf[i] /= N_;
    }
}

FFTW_SERIAL::~FFTW_SERIAL() {
    if (dtype_ == "float64") {
        fftw_destroy_plan(static_cast<fftw_plan>(plan_r2c_));
        fftw_destroy_plan(static_cast<fftw_plan>(plan_c2r_));
    } else if (dtype_ == "float32") {
        fftwf_destroy_plan(static_cast<fftwf_plan>(plan_r2c_));
        fftwf_destroy_plan(static_cast<fftwf_plan>(plan_c2r_));
    }
}