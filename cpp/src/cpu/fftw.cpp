/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/cpu/fftw.hpp"

// Helper: Geometry Calculation
struct GuruDims {
    std::vector<fftw_iodim> dims;
    std::vector<fftw_iodim> howmany_dims;
    long long N = 1;
};

static GuruDims compute_guru_dims(const std::vector<int>& shape,
                                  const std::vector<int>& axes,
                                  const std::vector<int>& s_in,
                                  const std::vector<int>& s_out)
{
    GuruDims g;
    std::set<int> axes_set(axes.begin(), axes.end());

    // Transform Dimensions
    for(int ax : axes) {
        fftw_iodim d;
        d.n = shape[ax];
        d.is = s_in[ax];
        d.os = s_out[ax];
        g.dims.push_back(d);
        g.N *= shape[ax];
    }

    // Batch Dimensions
    for(size_t i=0; i<shape.size(); ++i) {
        if(axes_set.find(i) == axes_set.end()) {
            fftw_iodim d;
            d.n = shape[i];
            d.is = s_in[i];
            d.os = s_out[i];
            g.howmany_dims.push_back(d);
        }
    }
    return g;
}

// Helper: Plan Cache
// Key: <is_double, algorithm_type, direction, flags, is_inplace, dims_flattened, howmany_flattened>
// algorithm_type: 0=C2C, 1=R2C, 2=C2R
using PlanKey = std::tuple<bool, int, int, unsigned, bool, std::vector<int>, std::vector<int>>;

class FFTWPlanCache {
    std::map<PlanKey, void*> cache_;
    std::mutex mtx_;

    // Helper to flatten iodims for the key
    std::vector<int> flatten_iodims(const std::vector<fftw_iodim>& dims) {
        std::vector<int> out;
        out.reserve(dims.size() * 3);
        for(const auto& d : dims) {
            out.push_back(d.n);
            out.push_back(d.is);
            out.push_back(d.os);
        }
        return out;
    }

public:
    static FFTWPlanCache& instance() {
        static FFTWPlanCache s;
        return s;
    }

    // Creator is a lambda that calls the actual FFTW plan function if cache miss
    template<typename Creator>
    void* get_or_create(bool is_double, int algo_type, int direction, unsigned flags,
                        bool is_inplace, const GuruDims& g, Creator&& creator)
    {
        PlanKey key = std::make_tuple(
            is_double, algo_type, direction, flags, is_inplace,
            flatten_iodims(g.dims), flatten_iodims(g.howmany_dims)
        );

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

class FFTW_LOCAL_C2C : public FFTBase {
    std::vector<int> shape_;
    ssize_t N_;
    std::string dtype_;
    void* plan_forward_;
    void* plan_backward_;

public:
    FFTW_LOCAL_C2C(const std::vector<int>& shape,
             py::array in, py::array out, std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), N_(1), dtype_(dtype), plan_forward_(nullptr), plan_backward_(nullptr)
    {
        for (int i : shape_) N_ *= i;
        void* i_p = in.mutable_data(); void* o_p = out.mutable_data();
        int ndim_ = shape_.size();

#ifdef ENABLE_FFTW_OMP
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
            if (dtype_ == "complex128")
                fftw_execute_dft((fftw_plan)plan_forward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_execute_dft((fftwf_plan)plan_forward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();

        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128")
                fftw_execute_dft((fftw_plan)plan_backward_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_execute_dft((fftwf_plan)plan_backward_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }

        scale_array(o_arr, 1.0/N_);
    }

    ~FFTW_LOCAL_C2C() {
        if(dtype_ == "complex128") {
            if(plan_forward_) fftw_destroy_plan((fftw_plan)plan_forward_);
            if(plan_backward_) fftw_destroy_plan((fftw_plan)plan_backward_);
        } else {
            if(plan_forward_) fftwf_destroy_plan((fftwf_plan)plan_forward_);
            if(plan_backward_) fftwf_destroy_plan((fftwf_plan)plan_backward_);
        }
    }
};

class FFTW_LOCAL_R2C_OutPlace : public FFTBase {
    std::vector<int> shape_;
    ssize_t N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_LOCAL_R2C_OutPlace(const std::vector<int>& shape,
                      py::array r, py::array c, std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int i:shape_) N_*=i;
        void* rp = r.mutable_data(); void* cp = c.mutable_data();
        int ndim_ = shape.size();

#ifdef ENABLE_FFTW_OMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if (dtype_ == "float64") {
            if (ndim_ == 1) {
                plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], (fftw_complex*)cp, (double*)rp, flags);
            } else if (ndim_ == 2) {
                plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0],shape_[1], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0],shape_[1], (fftw_complex*)cp, (double*)rp, flags);
            } else {
                plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (double*)rp, (fftw_complex*)cp, flags);
                plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftw_complex*)cp, (double*)rp, flags);
            }
        } else {
            if (ndim_ == 1) {
                plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], (float*)rp, (fftwf_complex*)cp, flags);
                plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], (fftwf_complex*)cp, (float*)rp, flags);
            } else if (ndim_ == 2) {
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

    ~FFTW_LOCAL_R2C_OutPlace() {
        if(dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_LOCAL_R2C_InPlace : public FFTBase {
    std::vector<int> shape_;
    ssize_t N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_LOCAL_R2C_InPlace(const std::vector<int>& shape, py::array data,
                     std::string dtype, int n_threads, unsigned flags)
        : shape_(shape), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for(int i:shape_) N_*=i;
        void* ptr = data.mutable_data();
        int ndim_ = shape.size();

#ifdef ENABLE_FFTW_OMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        if (dtype_ == "float64") {
            if (ndim_ == 1) {
                plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
                plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0],shape_[1], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0],shape_[1], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            } else {
                plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0],shape_[1],shape_[2], (double*)ptr, (fftw_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0],shape_[1],shape_[2], (fftw_complex*)ptr, (double*)ptr, FFTW_ESTIMATE);
            }
        } else {
            if (ndim_ == 1) {
                plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], (float*)ptr, (fftwf_complex*)ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], (fftwf_complex*)ptr, (float*)ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
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
            if(dtype_ == "float64")
                fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)p, (fftw_complex*)p);
            else
                fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)p, (fftwf_complex*)p);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array o_arr = out.cast<py::array>();

        void* p = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64")
                fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)p, (double*)p);
            else
                fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)p, (float*)p);
        }

        scale_array(o_arr, 1.0/N_);
    }

    ~FFTW_LOCAL_R2C_InPlace() {
        if(dtype_ == "float64") {
            if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_);
            if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_);
        } else {
            if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_);
            if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_);
        }
    }
};

class FFTW_LOCAL_GURU_C2C : public FFTBase {
    std::string dtype_;
    void* plan_fwd_ = nullptr;
    void* plan_bwd_ = nullptr;
    double scale_;

public:
    FFTW_LOCAL_GURU_C2C(const std::vector<int>& shape, const std::vector<int>& axes,
                  py::array in, py::array out, std::string dtype, int n_threads, unsigned flags)
        : dtype_(dtype)
    {
#ifdef ENABLE_FFTW_OMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        std::vector<int> s_in = get_array_strides(in);
        std::vector<int> s_out = get_array_strides(out);

        GuruDims g = compute_guru_dims(shape, axes, s_in, s_out);
        scale_ = 1.0 / (double)g.N;

        void* ip = in.mutable_data();
        void* op = out.mutable_data();
        bool is_double = (dtype_ == "complex128");
        bool is_inplace = (ip == op);

        plan_fwd_ = FFTWPlanCache::instance().get_or_create(
            is_double, 0 /*C2C*/, FFTW_FORWARD, flags, is_inplace, g,
            [&]() -> void* {
                if (is_double)
                    return fftw_plan_guru_dft(g.dims.size(), g.dims.data(), g.howmany_dims.size(), g.howmany_dims.data(), (fftw_complex*)ip, (fftw_complex*)op, FFTW_FORWARD, flags);
                else
                    return fftwf_plan_guru_dft(g.dims.size(), (fftwf_iodim*)g.dims.data(), g.howmany_dims.size(), (fftwf_iodim*)g.howmany_dims.data(), (fftwf_complex*)ip, (fftwf_complex*)op, FFTW_FORWARD, flags);
            }
        );

        plan_bwd_ = FFTWPlanCache::instance().get_or_create(
            is_double, 0 /*C2C*/, FFTW_BACKWARD, flags, is_inplace, g,
            [&]() -> void* {
                if (is_double)
                    return fftw_plan_guru_dft(g.dims.size(), g.dims.data(), g.howmany_dims.size(), g.howmany_dims.data(), (fftw_complex*)ip, (fftw_complex*)op, FFTW_BACKWARD, flags);
                else
                    return fftwf_plan_guru_dft(g.dims.size(), (fftwf_iodim*)g.dims.data(), g.howmany_dims.size(), (fftwf_iodim*)g.howmany_dims.data(), (fftwf_complex*)ip, (fftwf_complex*)op, FFTW_BACKWARD, flags);
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
            if(dtype_ == "complex128")
                fftw_execute_dft((fftw_plan)plan_fwd_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_execute_dft((fftwf_plan)plan_fwd_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "complex128")
                fftw_execute_dft((fftw_plan)plan_bwd_, (fftw_complex*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_execute_dft((fftwf_plan)plan_bwd_, (fftwf_complex*)i_ptr, (fftwf_complex*)o_ptr);
        }
        scale_array(o_arr, scale_);
    }

    ~FFTW_LOCAL_GURU_C2C() = default;
};

class FFTW_LOCAL_Guru_R2C : public FFTBase {
    std::string dtype_;
    void* plan_r2c_ = nullptr;
    void* plan_c2r_ = nullptr;
    double scale_;

public:
    FFTW_LOCAL_Guru_R2C(const std::vector<int>& shape, const std::vector<int>& axes,
                  py::array in, py::array out, std::string dtype, int n_threads, unsigned flags)
        : dtype_(dtype)
    {
#ifdef ENABLE_FFTW_OMP
        fftw_plan_with_nthreads(n_threads);
        fftwf_plan_with_nthreads(n_threads);
#endif

        std::vector<int> s_in = get_array_strides(in);
        std::vector<int> s_out = get_array_strides(out);

        GuruDims gf = compute_guru_dims(shape, axes, s_in, s_out);
        scale_ = 1.0 / (double)gf.N;

        GuruDims gb = compute_guru_dims(shape, axes, s_out, s_in);

        void* rp = in.mutable_data();
        void* cp = out.mutable_data();
        bool is_double = (dtype_ == "float64");
        bool is_inplace = (rp == cp);

        plan_r2c_ = FFTWPlanCache::instance().get_or_create(
            is_double, 1 /*R2C*/, FFTW_FORWARD, flags, is_inplace, gf,
            [&]() -> void* {
                if(is_double)
                    return fftw_plan_guru_dft_r2c(gf.dims.size(), gf.dims.data(), gf.howmany_dims.size(), gf.howmany_dims.data(), (double*)rp, (fftw_complex*)cp, flags);
                else
                    return fftwf_plan_guru_dft_r2c(gf.dims.size(), (fftwf_iodim*)gf.dims.data(), gf.howmany_dims.size(), (fftwf_iodim*)gf.howmany_dims.data(), (float*)rp, (fftwf_complex*)cp, flags);
            }
        );

        plan_c2r_ = FFTWPlanCache::instance().get_or_create(
            is_double, 2 /*C2R*/, FFTW_BACKWARD, flags, is_inplace, gb,
            [&]() -> void* {
                if(is_double)
                    return fftw_plan_guru_dft_c2r(gb.dims.size(), gb.dims.data(), gb.howmany_dims.size(), gb.howmany_dims.data(), (fftw_complex*)cp, (double*)rp, flags);
                else
                    return fftwf_plan_guru_dft_c2r(gb.dims.size(), (fftwf_iodim*)gb.dims.data(), gb.howmany_dims.size(), (fftwf_iodim*)gb.howmany_dims.data(), (fftwf_complex*)cp, (float*)rp, flags);
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
            if(dtype_ == "float64")
                fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)i_ptr, (fftw_complex*)o_ptr);
            else
                fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)i_ptr, (fftwf_complex*)o_ptr);
        }
    }

    void backward(py::object in, py::object out) override {
        py::array i_arr = in.cast<py::array>();
        py::array o_arr = out.cast<py::array>();
        void* i_ptr = i_arr.mutable_data();
        void* o_ptr = o_arr.mutable_data();

        {
            py::gil_scoped_release release;
            if(dtype_ == "float64")
                fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)i_ptr, (double*)o_ptr);
            else
                fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)i_ptr, (float*)o_ptr);
        }
        scale_array(o_arr, scale_);
    }

    ~FFTW_LOCAL_Guru_R2C() = default;
};

FFTW_LOCAL::FFTW_LOCAL(const std::vector<int>& shape,
                         const std::vector<int>& axes,
                         py::array real_in, py::array complex_out,
                         const std::string& dtype, int n_threads, unsigned flags)
{
#ifdef ENABLE_FFTW_OMP
    static bool initialized = false;
    if (!initialized) {
        fftw_init_threads();
        fftwf_init_threads();
        initialized = true;
    }
#endif

    std::vector<int> active_axes = axes;
    if (active_axes.empty()) {
        active_axes.resize(shape.size());
        std::iota(active_axes.begin(), active_axes.end(), 0);
    }

    if (dtype == "complex128" || dtype == "complex64")
        impl_ = std::make_unique<FFTW_LOCAL_GURU_C2C>(shape, active_axes, real_in, complex_out, dtype, n_threads, flags);
    else
        impl_ = std::make_unique<FFTW_LOCAL_Guru_R2C>(shape, active_axes, real_in, complex_out, dtype, n_threads, flags);
}

void FFTW_LOCAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void FFTW_LOCAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
FFTW_LOCAL::~FFTW_LOCAL() = default;
