/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/cpu/fftw_mpi.hpp"

// Helper: MPI Plan Cache
// Key: <comm_handle, is_double, algo_type, direction, flags, is_inplace, shape>
// algo_type: 0=C2C, 1=R2C, 2=C2R
using DistPlanKey = std::tuple<int, bool, int, int, unsigned, bool, std::vector<ptrdiff_t>>;

class FFTWDistPlanCache {
    std::map<DistPlanKey, void*> cache_;
    std::mutex mtx_;

public:
    static FFTWDistPlanCache& instance() {
        static FFTWDistPlanCache s;
        return s;
    }

    template<typename Creator>
    void* get_or_create(int comm_handle, bool is_double, int algo_type, int direction,
                        unsigned flags, bool is_inplace, const std::vector<ptrdiff_t>& shape,
                        Creator&& creator)
    {
        DistPlanKey key = std::make_tuple(comm_handle, is_double, algo_type, direction, flags, is_inplace, shape);

        std::lock_guard<std::mutex> lock(mtx_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        // Cache miss: Create plan
        void* plan = creator();
        if (plan) {
            cache_[key] = plan;
        }
        return plan;
    }
};

// STATIC HELPER: Layout Calculator
std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
FFTW_DIST::get_local_info(const std::vector<int>& global_shape, int comm_handle, bool r2c) {

    MPI_Comm comm = get_mpi_comm(comm_handle);
    static bool initialized = false;
    if (!initialized) { fftw_mpi_init(); initialized = true; }

    int ndim = global_shape.size();
    std::vector<ptrdiff_t> n(global_shape.begin(), global_shape.end());
    ptrdiff_t loc_n0, loc_0_start;

    // Generic N-Dimensional Local Size Calculator
    fftw_mpi_local_size(ndim, n.data(), comm, &loc_n0, &loc_0_start);

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

class FFTW_DIST_C2C : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_forward_;
    void* plan_backward_;

public:
    FFTW_DIST_C2C(int ndim, const std::vector<int>& shape,
                 py::array in, py::array out, std::string dtype, MPI_Comm comm)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype),
          plan_forward_(nullptr), plan_backward_(nullptr)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* i_p = in.mutable_data();
        void* o_p = out.mutable_data();

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

    void forward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128")
                fftw_mpi_execute_dft((fftw_plan)plan_forward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_mpi_execute_dft((fftwf_plan)plan_forward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();
        ssize_t size = o_arr.size();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") {
                fftw_mpi_execute_dft((fftw_plan)plan_backward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);

                double* buf = (double*)o_ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            } else {
                fftwf_mpi_execute_dft((fftwf_plan)plan_backward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);

                float* buf = (float*)o_ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            }
        }
    }

    ~FFTW_DIST_C2C() {
        if (dtype_ == "complex128") {
            if(plan_forward_) fftw_destroy_plan((fftw_plan)plan_forward_);
            if(plan_backward_) fftw_destroy_plan((fftw_plan)plan_backward_);
        } else {
            if(plan_forward_) fftwf_destroy_plan((fftwf_plan)plan_forward_);
            if(plan_backward_) fftwf_destroy_plan((fftwf_plan)plan_backward_);
        }
    }
};

class FFTW_DIST_R2C_InPlace : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_DIST_R2C_InPlace(int ndim, const std::vector<int>& shape,
                         py::array in, py::array out, std::string dtype, MPI_Comm comm)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype),
          plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* ptr = in.mutable_data();
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

        if (!plan_r2c_ || !plan_c2r_) throw std::runtime_error("FFTW MPI R2C In-Place: Hardcoded Plan Failed");
    }

    void forward(py::object in, py::object out) override {
        py::array arr = in.cast<py::array>();

        void* ptr = arr.mutable_data();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64")
                fftw_mpi_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)ptr, (fftw_complex*)ptr);
            else
                fftwf_mpi_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)ptr, (fftwf_complex*)ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array arr = in.cast<py::array>();

        void* ptr = arr.mutable_data();
        ssize_t size = arr.size();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") {
                fftw_mpi_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)ptr, (double*)ptr);

                double* buf = (double*)ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            } else {
                fftwf_mpi_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)ptr, (float*)ptr);

                float* buf = (float*)ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            }
        }
    }

    ~FFTW_DIST_R2C_InPlace() {
        if (dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_DIST_R2C_OutPlace : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_DIST_R2C_OutPlace(int ndim, const std::vector<int>& shape,
                          py::array in, py::array out, std::string dtype, MPI_Comm comm)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* rp = in.mutable_data();
        void* cp = out.mutable_data();
        unsigned flags = FFTW_ESTIMATE;

        if (dtype_ == "float64") {
            double* r = (double*)rp; fftw_complex* c = (fftw_complex*)cp;
            if (ndim == 3) {
                plan_r2c_ = fftw_mpi_plan_dft_r2c_3d(n[0], n[1], n[2], r, c, comm, flags);
                plan_c2r_ = fftw_mpi_plan_dft_c2r_3d(n[0], n[1], n[2], c, r, comm, flags);
            } else {
                plan_r2c_ = fftw_mpi_plan_dft_r2c_2d(n[0], n[1], r, c, comm, flags);
                plan_c2r_ = fftw_mpi_plan_dft_c2r_2d(n[0], n[1], c, r, comm, flags);
            }
        } else {
            float* r = (float*)rp; fftwf_complex* c = (fftwf_complex*)cp;
            if (ndim == 3) {
                plan_r2c_ = fftwf_mpi_plan_dft_r2c_3d(n[0], n[1], n[2], r, c, comm, flags);
                plan_c2r_ = fftwf_mpi_plan_dft_c2r_3d(n[0], n[1], n[2], c, r, comm, flags);
            } else {
                plan_r2c_ = fftwf_mpi_plan_dft_r2c_2d(n[0], n[1], r, c, comm, flags);
                plan_c2r_ = fftwf_mpi_plan_dft_c2r_2d(n[0], n[1], c, r, comm, flags);
            }
        }
        if(!plan_r2c_ || !plan_c2r_) throw std::runtime_error("FFTW MPI R2C Out-of-Place: Hardcoded Plan Failed");
    }

    void forward(py::object in, py::object out) override {
        void* r = in.cast<py::array>().mutable_data();
        void* c = out.cast<py::array>().mutable_data();

        py::gil_scoped_release release;
        if (dtype_ == "float64")
            fftw_mpi_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)r, (fftw_complex*)c);
        else
            fftwf_mpi_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)r, (fftwf_complex*)c);
    }

    void backward(py::object in, py::object out) override {
        void* c = in.cast<py::array>().mutable_data();
        py::array r_arr = out.cast<py::array>();

        void* r = r_arr.mutable_data();
        ssize_t size = r_arr.size();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") {
                fftw_mpi_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)c, (double*)r);
                double* buf = (double*)r;
                for(ssize_t k=0; k<size; ++k) buf[k] /= global_N_;
            } else {
                fftwf_mpi_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)c, (float*)r);
                float* buf = (float*)r;
                for(ssize_t k=0; k<size; ++k) buf[k] /= global_N_;
            }
        }
    }

    ~FFTW_DIST_R2C_OutPlace() {
        if (dtype_ == "float64") {
            fftw_destroy_plan((fftw_plan)plan_r2c_);
            fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_DIST_C2C_Generic : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    void* plan_forward_ = nullptr;
    void* plan_backward_ = nullptr;

public:
    FFTW_DIST_C2C_Generic(int ndim, const std::vector<int>& shape,
                 py::array in, py::array out, std::string dtype, MPI_Comm comm, int comm_handle)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* i_p = in.mutable_data();
        void* o_p = out.mutable_data();
        unsigned flags = FFTW_ESTIMATE;
        bool is_double = (dtype_ == "complex128");
        bool is_inplace = (i_p == o_p);

        plan_forward_ = FFTWDistPlanCache::instance().get_or_create(
            comm_handle, is_double, 0 /*C2C*/, FFTW_FORWARD, flags, is_inplace, n,
            [&]() -> void* {
                if (is_double)
                    return fftw_mpi_plan_dft(ndim, n.data(), (fftw_complex*)i_p, (fftw_complex*)o_p, comm, FFTW_FORWARD, flags);
                else
                    return fftwf_mpi_plan_dft(ndim, n.data(), (fftwf_complex*)i_p, (fftwf_complex*)o_p, comm, FFTW_FORWARD, flags);
            }
        );

        plan_backward_ = FFTWDistPlanCache::instance().get_or_create(
            comm_handle, is_double, 0 /*C2C*/, FFTW_BACKWARD, flags, is_inplace, n,
            [&]() -> void* {
                if (is_double)
                    return fftw_mpi_plan_dft(ndim, n.data(), (fftw_complex*)i_p, (fftw_complex*)o_p, comm, FFTW_BACKWARD, flags);
                else
                    return fftwf_mpi_plan_dft(ndim, n.data(), (fftwf_complex*)i_p, (fftwf_complex*)o_p, comm, FFTW_BACKWARD, flags);
            }
        );
    }

    void forward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128")
                fftw_mpi_execute_dft((fftw_plan)plan_forward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_mpi_execute_dft((fftwf_plan)plan_forward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();
        ssize_t size = o_arr.size();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") {
                fftw_mpi_execute_dft((fftw_plan)plan_backward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
                double* buf = (double*)o_ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            } else {
                fftwf_mpi_execute_dft((fftwf_plan)plan_backward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
                float* buf = (float*)o_ptr;
                for(ssize_t i=0; i<size*2; ++i) buf[i] /= global_N_;
            }
        }
    }
};

class FFTW_DIST_R2C_Generic : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t global_N_;
    std::string dtype_;
    bool is_inplace_;
    void* plan_r2c_ = nullptr;
    void* plan_c2r_ = nullptr;

public:
    FFTW_DIST_R2C_Generic(int ndim, const std::vector<int>& shape,
                         py::array in, py::array out, std::string dtype, MPI_Comm comm, int comm_handle)
        : shape_(shape), ndim_(ndim), global_N_(1), dtype_(dtype)
    {
        for(int s : shape_) global_N_ *= s;
        std::vector<ptrdiff_t> n(shape_.begin(), shape_.end());

        void* rp = in.mutable_data();
        void* cp = out.mutable_data();
        unsigned flags = FFTW_ESTIMATE;
        bool is_double = (dtype_ == "float64");

        is_inplace_ = (in.ptr() == out.ptr());

        plan_r2c_ = FFTWDistPlanCache::instance().get_or_create(
            comm_handle, is_double, 1 /*R2C*/, FFTW_FORWARD, flags, is_inplace_, n,
            [&]() -> void* {
                if (is_double)
                    return fftw_mpi_plan_dft_r2c(ndim, n.data(), (double*)rp, (fftw_complex*)cp, comm, flags);
                else
                    return fftwf_mpi_plan_dft_r2c(ndim, n.data(), (float*)rp, (fftwf_complex*)cp, comm, flags);
            }
        );

        plan_c2r_ = FFTWDistPlanCache::instance().get_or_create(
            comm_handle, is_double, 2 /*C2R*/, FFTW_BACKWARD, flags, is_inplace_, n,
            [&]() -> void* {
                if (is_double)
                    return fftw_mpi_plan_dft_c2r(ndim, n.data(), (fftw_complex*)cp, (double*)rp, comm, flags);
                else
                    return fftwf_mpi_plan_dft_c2r(ndim, n.data(), (fftwf_complex*)cp, (float*)rp, comm, flags);
            }
        );

        if (!plan_r2c_ || !plan_c2r_) throw std::runtime_error("R2C Generic Plan failed.");
    }

    void forward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* rp = i_arr.mutable_data();
        void* cp = o_arr.mutable_data();

        if (is_inplace_ && rp != cp) throw std::runtime_error("Generic R2C Configured In-Place but called Out-Place.");

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64")
                fftw_mpi_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)rp, (fftw_complex*)cp);
            else
                fftwf_mpi_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)rp, (fftwf_complex*)cp);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* cp = i_arr.mutable_data();
        void* rp = o_arr.mutable_data();
        ssize_t size = o_arr.size();

        if (is_inplace_ && rp != cp) throw std::runtime_error("Generic R2C Configured In-Place but called Out-Place.");

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") {
                fftw_mpi_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)cp, (double*)rp);
                double* buf = (double*)rp;
                for(ssize_t i=0; i<size; ++i) buf[i] /= global_N_;
            } else {
                fftwf_mpi_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)cp, (float*)rp);
                float* buf = (float*)rp;
                for(ssize_t i=0; i<size; ++i) buf[i] /= global_N_;
            }
        }
    }
};

FFTW_DIST::FFTW_DIST(const std::vector<int>& global_shape,
                   py::array in, py::array out,
                   const std::string& dtype, int comm_handle)
{
    static bool initialized = false;
    if (!initialized) { fftw_mpi_init(); initialized = true; }

    MPI_Comm comm = get_mpi_comm(comm_handle);
    int ndim = global_shape.size();
    bool use_hardcoded = (ndim == 2 || ndim == 3);

    if (dtype == "complex128" || dtype == "complex64") {
        impl_ = std::make_unique<FFTW_DIST_C2C_Generic>(ndim, global_shape, in, out, dtype, comm, comm_handle);
    } else {
        if (in.ptr() == out.ptr()) {
            if (use_hardcoded)
                impl_ = std::make_unique<FFTW_DIST_R2C_InPlace>(ndim, global_shape, in, out, dtype, comm);
            else
                impl_ = std::make_unique<FFTW_DIST_R2C_Generic>(ndim, global_shape, in, out, dtype, comm, comm_handle);
        } else {
            PyErr_SetString(PyExc_NotImplementedError, "R2C Out-of-Place not supported. Please use In-Place with a padded buffer.");
            throw pybind11::error_already_set();
            // if (use_hardcoded)
            //     impl_ = std::make_unique<FFTW_DIST_R2C_OutPlace>(ndim, global_shape, in, out, dtype, comm);
            // else
            //     impl_ = std::make_unique<FFTW_DIST_R2C_Generic>(ndim, global_shape, in, out, dtype, comm, comm_handle);
        }
    }
}

void FFTW_DIST::forward(py::object in, py::object out) { impl_->forward(in, out); }
void FFTW_DIST::backward(py::object in, py::object out) { impl_->backward(in, out); }
FFTW_DIST::~FFTW_DIST() = default;
