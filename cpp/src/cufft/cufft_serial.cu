/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/cufft_serial.cuh"

// Helper Macros
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
}
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFT Error: " + std::to_string(err)); \
}

// Helper: Get strides from CuPy object in elements
std::vector<int> get_cupy_strides(py::object arr) {
    std::vector<int> strides;
    py::tuple s = arr.attr("strides");
    int itemsize = arr.attr("itemsize").cast<int>();
    for (size_t i = 0; i < s.size(); ++i) {
        strides.push_back(s[i].cast<int>() / itemsize);
    }
    return strides;
}

static bool is_contiguous_axes(int ndim, const std::vector<int>& axes) {
    if (axes.empty()) return false;
    for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] != (ndim - axes.size() + i)) return false;
    }
    return true;
}

class CUFFT_C2C_Impl : public FFTBase {
    cufftHandle plan_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;
public:
    CUFFT_C2C_Impl(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim) {
        for (int s : shape) N_ *= s;

        CUFFT_CHECK(cufftCreate(&plan_));
        cufftType t_c2c = (dtype_ == "complex128") ? CUFFT_Z2Z : CUFFT_C2C;
        std::vector<int> n(shape.begin(), shape.end());
        CUFFT_CHECK(cufftPlanMany(&plan_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_c2c, 1));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD))
            else CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "complex128") CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE))
            else CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
    }

    ~CUFFT_C2C_Impl() {
        cufftDestroy(plan_);
    }
};

class CUFFT_R2C_Impl : public FFTBase {
    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;
public:
    CUFFT_R2C_Impl(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim) {
        for (int s : shape) N_ *= s;

        CUFFT_CHECK(cufftCreate(&plan_r2c_));
        CUFFT_CHECK(cufftCreate(&plan_c2r_));
        cufftType t_r2c = (dtype_ == "float64") ? CUFFT_D2Z : CUFFT_R2C;
        cufftType t_c2r = (dtype_ == "float64") ? CUFFT_Z2D : CUFFT_C2R;
        std::vector<int> n(shape.begin(), shape.end());
        CUFFT_CHECK(cufftPlanMany(&plan_r2c_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_r2c, 1));
        CUFFT_CHECK(cufftPlanMany(&plan_c2r_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_c2r, 1));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t r_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t c_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)r_ptr, (cufftDoubleComplex*)c_ptr))
            else CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)r_ptr, (cufftComplex*)c_ptr));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t c_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t r_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            if (dtype_ == "float64") CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)c_ptr, (cufftDoubleReal*)r_ptr))
            else CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)c_ptr, (cufftReal*)r_ptr));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
    }

    ~CUFFT_R2C_Impl() {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
};

class CUFFT_Generic : public FFTBase {
    int ndim_;
    std::vector<int> shape_;
    std::vector<int> axes_;
    std::string dtype_;

    cufftHandle plan_ = 0;
    bool plan_valid_ = false;

    struct PlanConfig {
        std::vector<int> n;
        int batch = 1;
        int istride = 1, idist = 1;
        std::vector<int> inembed;
        int ostride = 1, odist = 1;
        std::vector<int> onembed;
        cufftType type;
        bool is_inplace = false;
        long long logical_size = 1;
    } current_config_;

public:
    CUFFT_Generic(const std::vector<int>& shape, const std::vector<int>& axes, const std::string& dtype)
        : shape_(shape), axes_(axes), dtype_(dtype)
    {
        std::sort(axes_.begin(), axes_.end());
        ndim_ = axes_.size();
    }

    ~CUFFT_Generic() {
        if (plan_valid_) cufftDestroy(plan_);
    }

    bool check_contiguous_axes() {
        if (axes_.empty()) return true;
        for (size_t i = 0; i < axes_.size() - 1; ++i) {
            if (axes_[i+1] != axes_[i] + 1) return false;
        }
        return true;
    }

    void ensure_plan(py::object in, py::object out, cufftType requested_type) {
        if (!check_contiguous_axes()) {
            throw std::runtime_error("cuFFT Generic: Transform axes must be contiguous indices (e.g., [0,1] or [1,2]). Split axes (e.g. [0,2]) are not supported.");
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
            // Outer Batching (Batch dims ... Transform dims)
            new_config.batch = 1;
            for(int i=0; i < axes_.front(); ++i) new_config.batch *= shape_[i];

            // Transform Strides: The stride of the LAST axis
            new_config.istride = in_strides.back();
            new_config.ostride = out_strides.back();

            // Distance: The stride of the LAST BATCH axis (axis immediately before transform)
            if (new_config.batch > 1) {
                new_config.idist = in_strides[axes_.front() - 1];
                new_config.odist = out_strides[axes_.front() - 1];
            } else {
                new_config.idist = total_transform_size;
                new_config.odist = total_transform_size;
            }

        } else if (is_inner_batch) {
            // Inner Batching (Transform dims ... Batch dims)
            new_config.batch = 1;
            for(size_t i = axes_.back() + 1; i < full_rank; ++i) new_config.batch *= shape_[i];

            // Transform Strides: Stride of the last transform axis (which is NOT 1 here!)
            new_config.istride = in_strides[axes_.back()];
            new_config.ostride = out_strides[axes_.back()];

            // Distance: Stride of the FIRST batch axis (immediately after transform)
            new_config.idist = in_strides[axes_.back() + 1];
            new_config.odist = out_strides[axes_.back() + 1];

        } else {
            // Middle Batching (Batch ... Transform ... Batch) - Not Supported
            throw std::runtime_error("cuFFT Generic: Transform axes must be at the start or end of the array. Middle transforms (e.g. shape (A,B,C), axis B) with surrounding batches are not supported.");
        }

        // Set Embed (Must match transform dims)
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
            if (plan_valid_) cufftDestroy(plan_);
            CUFFT_CHECK(cufftCreate(&plan_));
            CUFFT_CHECK(cufftPlanMany(&plan_,
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
        cufftType type = is_c2c ? (is_double ? CUFFT_Z2Z : CUFFT_C2C)
                                : (is_double ? CUFFT_D2Z : CUFFT_R2C);

        ensure_plan(in, out, type);

        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            switch (current_config_.type) {
                case CUFFT_Z2Z: CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD)); break;
                case CUFFT_C2C: CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD)); break;
                case CUFFT_D2Z: CUFFT_CHECK(cufftExecD2Z(plan_, (cufftDoubleReal*)i_ptr, (cufftDoubleComplex*)o_ptr)); break;
                case CUFFT_R2C: CUFFT_CHECK(cufftExecR2C(plan_, (cufftReal*)i_ptr, (cufftComplex*)o_ptr)); break;
                default: break;
            }
        }
    }

    void backward(py::object in, py::object out) override {
        bool is_double = (dtype_ == "float64" || dtype_ == "complex128");
        bool is_c2c = (dtype_ == "complex64" || dtype_ == "complex128");
        cufftType type = is_c2c ? (is_double ? CUFFT_Z2Z : CUFFT_C2C)
                                : (is_double ? CUFFT_Z2D : CUFFT_C2R);

        ensure_plan(in, out, type);

        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        {
            py::gil_scoped_release release;
            switch (current_config_.type) {
                case CUFFT_Z2Z: CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE)); break;
                case CUFFT_C2C: CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE)); break;
                case CUFFT_Z2D: CUFFT_CHECK(cufftExecZ2D(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleReal*)o_ptr)); break;
                case CUFFT_C2R: CUFFT_CHECK(cufftExecC2R(plan_, (cufftComplex*)i_ptr, (cufftReal*)o_ptr)); break;
                default: break;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        double factor = 1.0 / (double)current_config_.logical_size;
        out.attr("__imul__")(factor);
    }
};

CUFFT_SERIAL::CUFFT_SERIAL(const std::vector<int>& shape,
                           const std::vector<int>& axes, const std::string& dtype)
{
    bool use_hardcoded = false;
    if (axes.empty()) {
        use_hardcoded = true;
    } else {
        if (axes.size() == shape.size()) {
            // Check if axes are [0, 1, 2...]
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
            impl_ = std::make_unique<CUFFT_C2C_Impl>(ndim, shape, dtype);
        else
            impl_ = std::make_unique<CUFFT_R2C_Impl>(ndim, shape, dtype);
    } else {
        // Fallback to Generic for partial transforms or permutations
        impl_ = std::make_unique<CUFFT_Generic>(shape, axes, dtype);
    }
}

void CUFFT_SERIAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void CUFFT_SERIAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
CUFFT_SERIAL::~CUFFT_SERIAL() = default;
