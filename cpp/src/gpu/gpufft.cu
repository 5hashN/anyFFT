/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/gpu/gpufft.cuh"

// Helper Macros
#define GPU_CHECK(call) { \
    gpufftError_t err = call; \
    if (err != GPUFFT_RUNTIME_SUCCESS) throw std::runtime_error("GPU Error: " + std::string(gpufftGetErrorString(err))); \
}
#define FFT_CHECK(call) { \
    gpufftResult err = call; \
    if (err != GPUFFT_SUCCESS) throw std::runtime_error("FFT Error: " + std::to_string(err)); \
}

// Helper: Get strides from CuPy object in elements
inline std::vector<int> get_cupy_strides(py::object arr) {
    std::vector<int> strides;
    py::tuple s = arr.attr("strides");
    int itemsize = arr.attr("itemsize").cast<int>();
    for (size_t i = 0; i < s.size(); ++i) {
        strides.push_back(s[i].cast<int>() / itemsize);
    }
    return strides;
}

class GPUFFT_LOCAL_C2C : public FFTBase {
    gpufftHandle plan_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;
public:
    GPUFFT_LOCAL_C2C(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim) {
        for (int s : shape) N_ *= s;

        FFT_CHECK(gpufftCreate(&plan_));
        gpufftType t_c2c = (dtype_ == "complex128") ? GPUFFT_Z2Z : GPUFFT_C2C;
        std::vector<int> n(shape.begin(), shape.end());
        FFT_CHECK(gpufftPlanMany(&plan_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_c2c, 1));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") FFT_CHECK(gpufftExecZ2Z(plan_, (gpufftDoubleComplex*)i_ptr, (gpufftDoubleComplex*)o_ptr, GPUFFT_FORWARD))
            else FFT_CHECK(gpufftExecC2C(plan_, (gpufftComplex*)i_ptr, (gpufftComplex*)o_ptr, GPUFFT_FORWARD));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") FFT_CHECK(gpufftExecZ2Z(plan_, (gpufftDoubleComplex*)i_ptr, (gpufftDoubleComplex*)o_ptr, GPUFFT_INVERSE))
            else FFT_CHECK(gpufftExecC2C(plan_, (gpufftComplex*)i_ptr, (gpufftComplex*)o_ptr, GPUFFT_INVERSE));
            GPU_CHECK(gpufftDeviceSynchronize());
        }

        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
    }

    ~GPUFFT_LOCAL_C2C() {
        gpufftDestroy(plan_);
    }
};

class GPUFFT_LOCAL_R2C : public FFTBase {
    gpufftHandle plan_r2c_;
    gpufftHandle plan_c2r_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;
public:
    GPUFFT_LOCAL_R2C(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim) {
        for (int s : shape) N_ *= s;

        FFT_CHECK(gpufftCreate(&plan_r2c_));
        FFT_CHECK(gpufftCreate(&plan_c2r_));
        gpufftType t_r2c = (dtype_ == "float64") ? GPUFFT_D2Z : GPUFFT_R2C;
        gpufftType t_c2r = (dtype_ == "float64") ? GPUFFT_Z2D : GPUFFT_C2R;
        std::vector<int> n(shape.begin(), shape.end());
        FFT_CHECK(gpufftPlanMany(&plan_r2c_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_r2c, 1));
        FFT_CHECK(gpufftPlanMany(&plan_c2r_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_c2r, 1));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t r_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t c_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") FFT_CHECK(gpufftExecD2Z(plan_r2c_, (gpufftDoubleReal*)r_ptr, (gpufftDoubleComplex*)c_ptr))
            else FFT_CHECK(gpufftExecR2C(plan_r2c_, (gpufftReal*)r_ptr, (gpufftComplex*)c_ptr));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t c_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t r_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") FFT_CHECK(gpufftExecZ2D(plan_c2r_, (gpufftDoubleComplex*)c_ptr, (gpufftDoubleReal*)r_ptr))
            else FFT_CHECK(gpufftExecC2R(plan_c2r_, (gpufftComplex*)c_ptr, (gpufftReal*)r_ptr));
            GPU_CHECK(gpufftDeviceSynchronize());
        }

        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
    }

    ~GPUFFT_LOCAL_R2C() {
        gpufftDestroy(plan_r2c_);
        gpufftDestroy(plan_c2r_);
    }
};

class GPUFFT_LOCAL_Generic : public FFTBase {
    int ndim_;
    std::vector<int> shape_;
    std::vector<int> axes_;
    std::string dtype_;

    gpufftHandle plan_ = 0;
    bool plan_valid_ = false;

    struct PlanConfig {
        std::vector<int> n;
        int batch = 1;
        int istride = 1, idist = 1;
        std::vector<int> inembed;
        int ostride = 1, odist = 1;
        std::vector<int> onembed;
        gpufftType type;
        bool is_inplace = false;
        long long logical_size = 1;
    } current_config_;

public:
    GPUFFT_LOCAL_Generic(const std::vector<int>& shape, const std::vector<int>& axes, const std::string& dtype)
        : shape_(shape), axes_(axes), dtype_(dtype)
    {
        std::sort(axes_.begin(), axes_.end());
        ndim_ = axes_.size();
    }

    ~GPUFFT_LOCAL_Generic() {
        if (plan_valid_) gpufftDestroy(plan_);
    }

    bool check_contiguous_axes() {
        if (axes_.empty()) return true;
        for (size_t i = 0; i < axes_.size() - 1; ++i) {
            if (axes_[i+1] != axes_[i] + 1) return false;
        }
        return true;
    }

    void ensure_plan(py::object in, py::object out, gpufftType requested_type) {
        if (!check_contiguous_axes()) {
            throw std::runtime_error("GPUFFT Generic: Transform axes must be contiguous indices (e.g., [0,1] or [1,2]). Split axes (e.g. [0,2]) are not supported.");
        }

        PlanConfig new_config;
        new_config.type = requested_type;

        std::vector<int> in_strides = get_cupy_strides(in);
        std::vector<int> out_strides = get_cupy_strides(out);

        uintptr_t in_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t out_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();
        new_config.is_inplace = (in_ptr == out_ptr);

        long long total_transform_size = 1;
        for (int ax : axes_) {
            new_config.n.push_back(shape_[ax]);
            total_transform_size *= shape_[ax];
        }
        new_config.logical_size = total_transform_size;
        int rank = new_config.n.size();

        int full_rank = shape_.size();
        bool is_outer_batch = (axes_.back() == full_rank - 1);
        bool is_inner_batch = (axes_.front() == 0);

        if (is_outer_batch) {
            new_config.batch = 1;
            for(int i=0; i < axes_.front(); ++i) new_config.batch *= shape_[i];

            new_config.istride = in_strides.back();
            new_config.ostride = out_strides.back();

            if (new_config.batch > 1) {
                new_config.idist = in_strides[axes_.front() - 1];
                new_config.odist = out_strides[axes_.front() - 1];
            } else {
                new_config.idist = total_transform_size;
                new_config.odist = total_transform_size;
            }

        } else if (is_inner_batch) {
            new_config.batch = 1;
            for(size_t i = axes_.back() + 1; i < full_rank; ++i) new_config.batch *= shape_[i];

            new_config.istride = in_strides[axes_.back()];
            new_config.ostride = out_strides[axes_.back()];

            new_config.idist = in_strides[axes_.back() + 1];
            new_config.odist = out_strides[axes_.back() + 1];

        } else {
            throw std::runtime_error("GPUFFT Generic: Transform axes must be at the start or end of the array. Middle transforms (e.g. shape (A,B,C), axis B) with surrounding batches are not supported.");
        }

        new_config.inembed = new_config.n;
        new_config.onembed = new_config.n;

        if (!plan_valid_ ||
            new_config.n != current_config_.n ||
            new_config.batch != current_config_.batch ||
            new_config.type != current_config_.type ||
            new_config.istride != current_config_.istride ||
            new_config.idist != current_config_.idist ||
            new_config.odist != current_config_.odist ||
            new_config.inembed != current_config_.inembed ||
            new_config.is_inplace != current_config_.is_inplace)
        {
            if (plan_valid_) gpufftDestroy(plan_);
            FFT_CHECK(gpufftCreate(&plan_));
            FFT_CHECK(gpufftPlanMany(&plan_,
                                      rank, new_config.n.data(),
                                      new_config.inembed.data(), new_config.istride, new_config.idist,
                                      new_config.onembed.data(), new_config.ostride, new_config.odist,
                                      new_config.type, new_config.batch));
            current_config_ = new_config;
            plan_valid_ = true;
        }
    }

    void forward(py::object in, py::object out) override {
        bool is_double = (dtype_ == "float64" || dtype_ == "complex128");
        bool is_c2c = (dtype_ == "complex64" || dtype_ == "complex128");
        gpufftType type = is_c2c ? (is_double ? GPUFFT_Z2Z : GPUFFT_C2C)
                                : (is_double ? GPUFFT_D2Z : GPUFFT_R2C);

        ensure_plan(in, out, type);

        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            switch (current_config_.type) {
                case GPUFFT_Z2Z: FFT_CHECK(gpufftExecZ2Z(plan_, (gpufftDoubleComplex*)i_ptr, (gpufftDoubleComplex*)o_ptr, GPUFFT_FORWARD)); break;
                case GPUFFT_C2C: FFT_CHECK(gpufftExecC2C(plan_, (gpufftComplex*)i_ptr, (gpufftComplex*)o_ptr, GPUFFT_FORWARD)); break;
                case GPUFFT_D2Z: FFT_CHECK(gpufftExecD2Z(plan_, (gpufftDoubleReal*)i_ptr, (gpufftDoubleComplex*)o_ptr)); break;
                case GPUFFT_R2C: FFT_CHECK(gpufftExecR2C(plan_, (gpufftReal*)i_ptr, (gpufftComplex*)o_ptr)); break;
                default: break;
            }
        }
    }

    void backward(py::object in, py::object out) override {
        bool is_double = (dtype_ == "float64" || dtype_ == "complex128");
        bool is_c2c = (dtype_ == "complex64" || dtype_ == "complex128");
        gpufftType type = is_c2c ? (is_double ? GPUFFT_Z2Z : GPUFFT_C2C)
                                : (is_double ? GPUFFT_Z2D : GPUFFT_C2R);

        ensure_plan(in, out, type);

        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            switch (current_config_.type) {
                case GPUFFT_Z2Z: FFT_CHECK(gpufftExecZ2Z(plan_, (gpufftDoubleComplex*)i_ptr, (gpufftDoubleComplex*)o_ptr, GPUFFT_INVERSE)); break;
                case GPUFFT_C2C: FFT_CHECK(gpufftExecC2C(plan_, (gpufftComplex*)i_ptr, (gpufftComplex*)o_ptr, GPUFFT_INVERSE)); break;
                case GPUFFT_Z2D: FFT_CHECK(gpufftExecZ2D(plan_, (gpufftDoubleComplex*)i_ptr, (gpufftDoubleReal*)o_ptr)); break;
                case GPUFFT_C2R: FFT_CHECK(gpufftExecC2R(plan_, (gpufftComplex*)i_ptr, (gpufftReal*)o_ptr)); break;
                default: break;
            }
            GPU_CHECK(gpufftDeviceSynchronize());
        }

        double factor = 1.0 / (double)current_config_.logical_size;
        out.attr("__imul__")(factor);
    }
};

GPUFFT_LOCAL::GPUFFT_LOCAL(const std::vector<int>& shape,
                           const std::vector<int>& axes, const std::string& dtype)
{
    bool use_hardcoded = false;
    if (axes.empty()) {
        use_hardcoded = true;
    } else {
        if (axes.size() == shape.size()) {
            bool sorted = true;
            std::vector<int> sorted_axes = axes;
            std::sort(sorted_axes.begin(), sorted_axes.end());
            for(size_t i=0; i<sorted_axes.size(); ++i) {
                if (sorted_axes[i] != i) { sorted = false; break; }
            }
            if (sorted) use_hardcoded = true;
        }
    }

    int ndim = shape.size();

    if (use_hardcoded) {
        if (dtype == "complex128" || dtype == "complex64")
            impl_ = std::make_unique<GPUFFT_LOCAL_C2C>(ndim, shape, dtype);
        else
            impl_ = std::make_unique<GPUFFT_LOCAL_R2C>(ndim, shape, dtype);
    } else {
        impl_ = std::make_unique<GPUFFT_LOCAL_Generic>(shape, axes, dtype);
    }
}

void GPUFFT_LOCAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void GPUFFT_LOCAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
GPUFFT_LOCAL::~GPUFFT_LOCAL() = default;
