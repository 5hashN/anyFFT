/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/fftw_serial.hpp"

class FFTW_C2C : public FFTBase {
    std::vector<int> shape_; int ndim_; ssize_t N_; std::string dtype_;
    void* plan_forward_; void* plan_backward_;

public:
    FFTW_C2C(int ndim, const std::vector<int>& shape, py::array in, py::array out,
             std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_forward_(nullptr), plan_backward_(nullptr)
    {
        for (int i : shape_) N_ *= i;
        void* i_p = in.mutable_data(); void* o_p = out.mutable_data();

#ifdef ENABLE_OPENMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if (dtype_ == "complex128") {
            if (ndim_ == 1) {
                plan_forward_ = fftw_plan_dft_1d(shape_[0], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftw_plan_dft_1d(shape_[0], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_BACKWARD, flags);
            } else if (ndim_ == 2) {
                plan_forward_ = fftw_plan_dft_2d(shape_[0], shape_[1], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftw_plan_dft_2d(shape_[0], shape_[1], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_BACKWARD, flags);
            } else if (ndim_ == 3) {
                plan_forward_ = fftw_plan_dft_3d(shape_[0], shape_[1], shape_[2], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftw_plan_dft_3d(shape_[0], shape_[1], shape_[2], (fftw_complex*)i_p, (fftw_complex*)o_p, FFTW_BACKWARD, flags);
            }
        } else {
            if (ndim_ == 1) {
                plan_forward_ = fftwf_plan_dft_1d(shape_[0], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftwf_plan_dft_1d(shape_[0], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_BACKWARD, flags);
            } else if (ndim_ == 2) {
                plan_forward_ = fftwf_plan_dft_2d(shape_[0], shape_[1], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftwf_plan_dft_2d(shape_[0], shape_[1], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_BACKWARD, flags);
            } else if (ndim_ == 3) {
                plan_forward_ = fftwf_plan_dft_3d(shape_[0], shape_[1], shape_[2], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_FORWARD, flags);
                plan_backward_ = fftwf_plan_dft_3d(shape_[0], shape_[1], shape_[2], (fftwf_complex*)i_p, (fftwf_complex*)o_p, FFTW_BACKWARD, flags);
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
            if (dtype_ == "complex128") fftw_execute_dft((fftw_plan)plan_forward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft((fftwf_plan)plan_forward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") fftw_execute_dft((fftw_plan)plan_backward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft((fftwf_plan)plan_backward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }

        scale_array(o_arr, 1.0/N_);
    }

    ~FFTW_C2C() {
        if(dtype_ == "complex128") {
            if(plan_forward_) fftw_destroy_plan((fftw_plan)plan_forward_);
            if(plan_backward_) fftw_destroy_plan((fftw_plan)plan_backward_);
        } else {
            if(plan_forward_) fftwf_destroy_plan((fftwf_plan)plan_forward_);
            if(plan_backward_) fftwf_destroy_plan((fftwf_plan)plan_backward_);
        }
    }
};

class FFTW_R2C_OutPlace : public FFTBase {
    std::vector<int> shape_; int ndim_; ssize_t N_; std::string dtype_;
    void* plan_r2c_; void* plan_c2r_;

public:
    FFTW_R2C_OutPlace(int ndim, const std::vector<int>& shape, py::array r, py::array c,
                      std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int i:shape_) N_*=i;
        void* rp = r.mutable_data(); void* cp = c.mutable_data();

#ifdef ENABLE_OPENMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if (dtype_ == "float64") {
            if (ndim_==1) {
                plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], (fftw_complex*)cp, (double*)rp, flags);
            } else if (ndim_==2) {
                plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0],shape_[1], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0],shape_[1], (fftw_complex*)cp, (double*)rp, flags);
            } else {
                plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftw_complex*)cp, (double*)rp, flags);
            }
        } else {
            if (ndim_==1) {
                plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], (float*)rp, (fftwf_complex*)cp, flags);
                plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], (fftwf_complex*)cp, (float*)rp, flags);
            } else if (ndim_==2) {
                plan_r2c_ = fftwf_plan_dft_r2c_2d(shape_[0],shape_[1], (float*)rp, (fftwf_complex*)cp, flags);
                plan_c2r_ = fftwf_plan_dft_c2r_2d(shape_[0],shape_[1], (fftwf_complex*)cp, (float*)rp, flags);
            } else {
                plan_r2c_ = fftwf_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (float*)rp, (fftwf_complex*)cp, flags);
                plan_c2r_ = fftwf_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftwf_complex*)cp, (float*)rp, flags);
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
            if(dtype_ == "float64") fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64") fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)i_ptr, (double*)o_ptr);
            else fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)i_ptr, (float*)o_ptr);
        }

        scale_array(o_arr, 1.0/N_);
    }

    ~FFTW_R2C_OutPlace() {
        if(dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_R2C_InPlace : public FFTBase {
    std::vector<int> shape_; int ndim_; ssize_t N_; std::string dtype_;
    void* plan_r2c_; void* plan_c2r_;

public:
    FFTW_R2C_InPlace(int ndim, const std::vector<int>& shape, py::array data,
                     std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int i:shape_) N_*=i;
        void* ptr = data.mutable_data();

#ifdef ENABLE_OPENMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if (dtype_ == "float64") {
            if (ndim_==1) {
                plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            } else if (ndim_==2) {
                plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0],shape_[1], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0],shape_[1], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            } else {
                plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            }
        } else {
            if (ndim_==1) {
                plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], (float*)ptr, (fftwf_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], (fftwf_complex*)ptr, (float*)ptr, FFTW_ESTIMATE);
            } else if (ndim_==2) {
                plan_r2c_ = fftwf_plan_dft_r2c_2d(shape_[0],shape_[1], (float*)ptr, (fftwf_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_2d(shape_[0],shape_[1], (fftwf_complex*)ptr, (float*)ptr, FFTW_ESTIMATE);
            } else {
                plan_r2c_ = fftwf_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (float*)ptr, (fftwf_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftwf_complex*)ptr, (float*)ptr, FFTW_ESTIMATE);
            }
        }
    }

    void forward(py::object in, py::object out) override {
        py::array arr = in.cast<py::array>();

        void* p = arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64") fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)p, (fftw_complex*)p);
            else fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)p, (fftwf_complex*)p);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array o_arr = out.cast<py::array>();

        void* p = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64") fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)p, (double*)p);
            else fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)p, (float*)p);
        }

        scale_array(o_arr, 1.0/N_);
    }

    ~FFTW_R2C_InPlace() {
        if(dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_Guru_C2C : public FFTBase {
    std::string dtype_;
    void* plan_fwd_ = nullptr; void* plan_bwd_ = nullptr;
    double scale_;

public:
    FFTW_Guru_C2C(const std::vector<int>& shape, const std::vector<int>& axes,
                  py::array in, py::array out, std::string dtype, int n_threads, unsigned flags)
        : dtype_(dtype)
    {
        std::vector<int> s_in = get_array_strides(in);
        std::vector<int> s_out = get_array_strides(out);
        std::set<int> axes_set(axes.begin(), axes.end());

        std::vector<fftw_iodim> dims, howmany_dims;
        long long N = 1;

        for(int ax : axes) {
            fftw_iodim d; d.n = shape[ax]; d.is = s_in[ax]; d.os = s_out[ax];
            dims.push_back(d); N *= shape[ax];
        }
        scale_ = 1.0 / (double)N;

        for(size_t i=0; i<shape.size(); ++i) {
            if(axes_set.find(i) == axes_set.end()) {
                fftw_iodim d; d.n = shape[i]; d.is = s_in[i]; d.os = s_out[i];
                howmany_dims.push_back(d);
            }
        }

        void* ip = in.mutable_data(); void* op = out.mutable_data();

#ifdef ENABLE_OPENMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if(dtype_ == "complex128") {
            plan_fwd_ = fftw_plan_guru_dft(dims.size(), dims.data(), howmany_dims.size(), howmany_dims.data(), (fftw_complex*)ip, (fftw_complex*)op, FFTW_FORWARD, flags);
            plan_bwd_ = fftw_plan_guru_dft(dims.size(), dims.data(), howmany_dims.size(), howmany_dims.data(), (fftw_complex*)ip, (fftw_complex*)op, FFTW_BACKWARD, flags);
        } else {
            plan_fwd_ = fftwf_plan_guru_dft(dims.size(), (fftwf_iodim*)dims.data(), howmany_dims.size(), (fftwf_iodim*)howmany_dims.data(), (fftwf_complex*)ip, (fftwf_complex*)op, FFTW_FORWARD, flags);
            plan_bwd_ = fftwf_plan_guru_dft(dims.size(), (fftwf_iodim*)dims.data(), howmany_dims.size(), (fftwf_iodim*)howmany_dims.data(), (fftwf_complex*)ip, (fftwf_complex*)op, FFTW_BACKWARD, flags);
        }
    }

    void forward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "complex128") fftw_execute_dft((fftw_plan)plan_fwd_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft((fftwf_plan)plan_fwd_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "complex128") fftw_execute_dft((fftw_plan)plan_bwd_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft((fftwf_plan)plan_bwd_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }

        scale_array(o_arr, scale_);
    }

    ~FFTW_Guru_C2C() {
        if(dtype_ == "complex128") {
            if(plan_fwd_) fftw_destroy_plan((fftw_plan)plan_fwd_);
            if(plan_bwd_) fftw_destroy_plan((fftw_plan)plan_bwd_);
        } else {
            if(plan_fwd_) fftwf_destroy_plan((fftwf_plan)plan_fwd_);
            if(plan_bwd_) fftwf_destroy_plan((fftwf_plan)plan_bwd_);
        }
    }
};

class FFTW_Guru_R2C : public FFTBase {
    std::string dtype_;
    void* plan_r2c_ = nullptr; void* plan_c2r_ = nullptr;
    double scale_;

public:
    FFTW_Guru_R2C(const std::vector<int>& shape, const std::vector<int>& axes,
                  py::array in, py::array out, std::string dtype, int n_threads, unsigned flags)
        : dtype_(dtype)
    {
        std::vector<int> s_in = get_array_strides(in);
        std::vector<int> s_out = get_array_strides(out);
        std::set<int> axes_set(axes.begin(), axes.end());

        std::vector<fftw_iodim> df, db, hf, hb;
        long long N = 1;

        for(int ax : axes) {
            fftw_iodim f, b;
            f.n = shape[ax]; f.is = s_in[ax]; f.os = s_out[ax];
            b.n = shape[ax]; b.is = s_out[ax]; b.os = s_in[ax]; // Swap Strides for BWD
            df.push_back(f); db.push_back(b);
            N *= shape[ax];
        }
        scale_ = 1.0 / (double)N;

        for(size_t i=0; i<shape.size(); ++i) {
            if(axes_set.find(i) == axes_set.end()) {
                fftw_iodim f, b;
                f.n = shape[i]; f.is = s_in[i]; f.os = s_out[i];
                b.n = shape[i]; b.is = s_out[i]; b.os = s_in[i]; // Swap Strides for BWD
                hf.push_back(f); hb.push_back(b);
            }
        }

        void* rp = in.mutable_data(); void* cp = out.mutable_data();

#ifdef ENABLE_OPENMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if(dtype_ == "float64") {
            plan_r2c_ = fftw_plan_guru_dft_r2c(df.size(), df.data(), hf.size(), hf.data(), (double*)rp, (fftw_complex*)cp, flags);
            plan_c2r_ = fftw_plan_guru_dft_c2r(db.size(), db.data(), hb.size(), hb.data(), (fftw_complex*)cp, (double*)rp, flags);
        } else {
            plan_r2c_ = fftwf_plan_guru_dft_r2c(df.size(), (fftwf_iodim*)df.data(), hf.size(), (fftwf_iodim*)hf.data(), (float*)rp, (fftwf_complex*)cp, flags);
            plan_c2r_ = fftwf_plan_guru_dft_c2r(db.size(), (fftwf_iodim*)db.data(), hb.size(), (fftwf_iodim*)hb.data(), (fftwf_complex*)cp, (float*)rp, flags);
        }
    }

    void forward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64") fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)i_ptr, (fftw_complex*)o_ptr);
            else fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64") fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)i_ptr, (double*)o_ptr);
            else fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)i_ptr, (float*)o_ptr);
        }

        scale_array(o_arr, scale_);
    }

    ~FFTW_Guru_R2C() {
        if(dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

FFTW_SERIAL::FFTW_SERIAL(int ndim, const std::vector<int>& shape,
                         const std::vector<int>& axes,
                         py::array real_in, py::array complex_out,
                         const std::string& dtype, int n_threads, unsigned flags)
{
#ifdef ENABLE_OPENMP
    static bool initialized = false;
    if (!initialized) {
        fftw_init_threads();
        fftwf_init_threads();
        initialized = true;
    }
#endif

    if (!axes.empty()) {
        if (dtype == "complex128" || dtype == "complex64") {
            impl_ = std::make_unique<FFTW_Guru_C2C>(shape, axes, real_in, complex_out, dtype, n_threads, flags);
        } else {
            impl_ = std::make_unique<FFTW_Guru_R2C>(shape, axes, real_in, complex_out, dtype, n_threads, flags);
        }
    } else {
        if (dtype == "complex128" || dtype == "complex64") {
            impl_ = std::make_unique<FFTW_C2C>(ndim, shape, real_in, complex_out, dtype, n_threads, flags);
        } else if (real_in.ptr() == complex_out.ptr()) {
            impl_ = std::make_unique<FFTW_R2C_InPlace>(ndim, shape, real_in, dtype, n_threads, flags);
        } else {
            impl_ = std::make_unique<FFTW_R2C_OutPlace>(ndim, shape, real_in, complex_out, dtype, n_threads, flags);
        }
    }
}

void FFTW_SERIAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void FFTW_SERIAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
FFTW_SERIAL::~FFTW_SERIAL() = default;