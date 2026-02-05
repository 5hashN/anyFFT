/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/fftw_mpi.hpp"

// STATIC HELPER: Layout Calculator
std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
FFTW_MPI::get_local_info(int ndim, const std::vector<int>& global_shape, int comm_handle, bool r2c) {

    MPI_Comm comm = get_mpi_comm(comm_handle);
    static bool initialized = false;
    if (!initialized) { fftw_mpi_init(); initialized = true; }

    ptrdiff_t loc_n0, loc_0_start;
    ptrdiff_t alloc_local;

    // Calculate Standard Slab Decomposition
    if (ndim == 3) {
        alloc_local = fftw_mpi_local_size_3d(global_shape[0], global_shape[1], global_shape[2], comm, &loc_n0, &loc_0_start);
    } else if (ndim == 2) {
        alloc_local = fftw_mpi_local_size_2d(global_shape[0], global_shape[1], comm, &loc_n0, &loc_0_start);
    } else {
        throw std::runtime_error("Only 2D and 3D transforms are supported.");
    }

    // Base Shape (Real / Logical Input)
    std::vector<long> shape(global_shape.begin(), global_shape.end());
    shape[0] = loc_n0;

    std::vector<long> start(ndim, 0);
    start[0] = loc_0_start;

    if (!r2c) {
        // C2C: Input and Output are identical
        return {shape, start, shape, start};
    } else {
        // R2C: Standard Slab
        // Input is the Real shape [loc_n0, ..., N_last]
        // Output is the Complex shape [loc_n0, ..., N_last/2 + 1]
        std::vector<long> out_shape = shape;
        out_shape.back() = global_shape.back() / 2 + 1;

        return {shape, start, out_shape, start};
    }
}

// MPI C2C In-Place and Out-of-Place
class FFTW_MPI_C2C : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_forward_;
    void* plan_backward_;

public:
    FFTW_MPI_C2C(int ndim, const std::vector<int>& shape,
                 py::array in_arr, py::array out_arr, MPI_Comm comm, std::string dtype)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype),
          plan_forward_(nullptr), plan_backward_(nullptr)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());
        void* i_p = in_arr.mutable_data();
        void* o_p = out_arr.mutable_data();

        unsigned flags = FFTW_ESTIMATE;

        if (dtype_ == "complex128") {
            fftw_complex* i = reinterpret_cast<fftw_complex*>(i_p);
            fftw_complex* o = reinterpret_cast<fftw_complex*>(o_p);
            if (ndim == 3) {
                plan_forward_ = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], i, o, comm, FFTW_FORWARD, flags);
                plan_backward_ = fftw_mpi_plan_dft_3d(n[0], n[1], n[2], i, o, comm, FFTW_BACKWARD, flags);
            } else {
                plan_forward_ = fftw_mpi_plan_dft_2d(n[0], n[1], i, o, comm, FFTW_FORWARD, flags);
                plan_backward_ = fftw_mpi_plan_dft_2d(n[0], n[1], i, o, comm, FFTW_BACKWARD, flags);
            }
        } else if (dtype_ == "complex64") {
            fftwf_complex* i = reinterpret_cast<fftwf_complex*>(i_p);
            fftwf_complex* o = reinterpret_cast<fftwf_complex*>(o_p);
            if (ndim == 3) {
                plan_forward_ = fftwf_mpi_plan_dft_3d(n[0], n[1], n[2], i, o, comm, FFTW_FORWARD, flags);
                plan_backward_ = fftwf_mpi_plan_dft_3d(n[0], n[1], n[2], i, o, comm, FFTW_BACKWARD, flags);
            } else {
                plan_forward_ = fftwf_mpi_plan_dft_2d(n[0], n[1], i, o, comm, FFTW_FORWARD, flags);
                plan_backward_ = fftwf_mpi_plan_dft_2d(n[0], n[1], i, o, comm, FFTW_BACKWARD, flags);
            }
        }
    }

    void forward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();
        if (dtype_ == "complex128")
            fftw_mpi_execute_dft((fftw_plan)plan_forward_, (fftw_complex*)in.mutable_data(), (fftw_complex*)out.mutable_data());
        else
            fftwf_mpi_execute_dft((fftwf_plan)plan_forward_, (fftwf_complex*)in.mutable_data(), (fftwf_complex*)out.mutable_data());
    }

    void backward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();

        if (dtype_ == "complex128") {
            fftw_mpi_execute_dft((fftw_plan)plan_backward_, (fftw_complex*)in.mutable_data(), (fftw_complex*)out.mutable_data());
            double* buf = (double*)out.mutable_data();
            for(ssize_t i=0; i<out.size()*2; ++i) buf[i] /= global_N_;
        } else {
            fftwf_mpi_execute_dft((fftwf_plan)plan_backward_, (fftwf_complex*)in.mutable_data(), (fftwf_complex*)out.mutable_data());
            float* buf = (float*)out.mutable_data();
            for(ssize_t i=0; i<out.size()*2; ++i) buf[i] /= global_N_;
        }
    }

    ~FFTW_MPI_C2C() {
        if (dtype_ == "complex128") { if(plan_forward_) fftw_destroy_plan((fftw_plan)plan_forward_); if(plan_backward_) fftw_destroy_plan((fftw_plan)plan_backward_); }
        else { if(plan_forward_) fftwf_destroy_plan((fftwf_plan)plan_forward_); if(plan_backward_) fftwf_destroy_plan((fftwf_plan)plan_backward_); }
    }
};

// MPI R2C In-Place Only
class FFTW_MPI_R2C_InPlace : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_MPI_R2C_InPlace(int ndim, const std::vector<int>& shape,
                         py::array in_arr, py::array out_arr, MPI_Comm comm, std::string dtype)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype),
          plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* ptr = in_arr.mutable_data();
        unsigned flags = FFTW_ESTIMATE;

        if (dtype_ == "float64") {
            double* rp = static_cast<double*>(ptr);
            fftw_complex* cp = reinterpret_cast<fftw_complex*>(ptr);

            if (ndim == 3) {
                plan_r2c_ = fftw_mpi_plan_dft_r2c_3d(n[0], n[1], n[2], rp, cp, comm, flags);
                plan_c2r_ = fftw_mpi_plan_dft_c2r_3d(n[0], n[1], n[2], cp, rp, comm, flags);
            } else {
                plan_r2c_ = fftw_mpi_plan_dft_r2c_2d(n[0], n[1], rp, cp, comm, flags);
                plan_c2r_ = fftw_mpi_plan_dft_c2r_2d(n[0], n[1], cp, rp, comm, flags);
            }
        }
        else if (dtype_ == "float32") {
            float* rp = static_cast<float*>(ptr);
            fftwf_complex* cp = reinterpret_cast<fftwf_complex*>(ptr);

            if (ndim == 3) {
                plan_r2c_ = fftwf_mpi_plan_dft_r2c_3d(n[0], n[1], n[2], rp, cp, comm, flags);
                plan_c2r_ = fftwf_mpi_plan_dft_c2r_3d(n[0], n[1], n[2], cp, rp, comm, flags);
            } else {
                plan_r2c_ = fftwf_mpi_plan_dft_r2c_2d(n[0], n[1], rp, cp, comm, flags);
                plan_c2r_ = fftwf_mpi_plan_dft_c2r_2d(n[0], n[1], cp, rp, comm, flags);
            }
        }

        if (!plan_r2c_ || !plan_c2r_) throw std::runtime_error("R2C Plan failed.");
    }

    void forward(py::object in_obj, py::object out_obj) override {
        py::array arr = in_obj.cast<py::array>();
        if (dtype_ == "float64")
            fftw_mpi_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)arr.mutable_data(), (fftw_complex*)arr.mutable_data());
        else
            fftwf_mpi_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)arr.mutable_data(), (fftwf_complex*)arr.mutable_data());
    }

    void backward(py::object in_obj, py::object out_obj) override {
        py::array arr = in_obj.cast<py::array>();

        if (dtype_ == "float64") {
            fftw_mpi_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)arr.mutable_data(), (double*)arr.mutable_data());
            // Normalize
            double* buf = (double*)arr.mutable_data();
            ssize_t total = arr.size() * 2;
            for(ssize_t i=0; i<total; ++i) buf[i] /= global_N_;
        } else {
            fftwf_mpi_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)arr.mutable_data(), (float*)arr.mutable_data());
            float* buf = (float*)arr.mutable_data();
            ssize_t total = arr.size() * 2;
            for(ssize_t i=0; i<total; ++i) buf[i] /= global_N_;
        }
    }

    ~FFTW_MPI_R2C_InPlace() {
        if (dtype_ == "float64") { if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_); if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_); }
        else { if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_); if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_); }
    }
};

// The Wrapper Constructor
FFTW_MPI::FFTW_MPI(int ndim, const std::vector<int>& global_shape,
                   py::array input, py::array output,
                   int comm_handle, const std::string& dtype)
{
    static bool initialized = false;
    if (!initialized) { fftw_mpi_init(); initialized = true; }
    MPI_Comm comm = get_mpi_comm(comm_handle);

    if (dtype == "complex128" || dtype == "complex64") {
        impl_ = std::make_unique<FFTW_MPI_C2C>(ndim, global_shape, input, output, comm, dtype);
    } else {
        if (input.ptr() != output.ptr()) {
            throw std::runtime_error("R2C Out-of-Place not supported. Please use In-Place with a padded buffer.");
        }
        impl_ = std::make_unique<FFTW_MPI_R2C_InPlace>(ndim, global_shape, input, output, comm, dtype);
    }
}

void FFTW_MPI::forward(py::object in, py::object out) { impl_->forward(in, out); }
void FFTW_MPI::backward(py::object in, py::object out) { impl_->backward(in, out); }
FFTW_MPI::~FFTW_MPI() = default;