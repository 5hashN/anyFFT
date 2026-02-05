#include "../includes/cufft_serial.cuh"

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

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD))
        } else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE))
        { else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE));
        }

        // Use CuPy for normalization to handle strides safely
        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
        CUDA_CHECK(cudaDeviceSynchronize());
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

        if (dtype_ == "float64") {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)r_ptr, (cufftDoubleComplex*)c_ptr))
        } else {
            CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)r_ptr, (cufftComplex*)c_ptr));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t c_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t r_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "float64") {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)c_ptr, (cufftDoubleReal*)r_ptr))
        } else {
            CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)c_ptr, (cufftReal*)r_ptr));
        }

        double factor = 1.0 / (double)N_;
        out.attr("__imul__")(factor);
        CUDA_CHECK(cudaDeviceSynchronize());
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

    void ensure_plan(py::object in, py::object out, cufftType requested_type) {
        // Explicit check for Rank > 3
        if (ndim_ > 3) {
            throw std::runtime_error("cuFFT Error: Rank > 3 is not supported (received rank " + std::to_string(ndim_) + ")");
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

        int full_rank = shape_.size();
        bool is_contiguous_suffix = is_contiguous_axes(full_rank, axes_);

        if (is_contiguous_suffix) {
            new_config.batch = 1;
            for (int i = 0; i < full_rank - ndim_; ++i) new_config.batch *= shape_[i];

            int last_dim_idx = axes_.back();
            new_config.istride = in_strides[last_dim_idx];
            new_config.ostride = out_strides[last_dim_idx];

            new_config.inembed = new_config.n;
            new_config.onembed = new_config.n;

            if (ndim_ > 1) {
                for (int i = ndim_ - 1; i > 0; --i) {
                    int dim_idx = axes_[i];
                    int prev_dim_idx = axes_[i-1];
                    // inembed[i] gets the physical size of dim i
                    new_config.inembed[i] = in_strides[prev_dim_idx] / in_strides[dim_idx];
                    new_config.onembed[i] = out_strides[prev_dim_idx] / out_strides[dim_idx];
                }
            }

            if (new_config.batch > 1) {
                int batch_dim = full_rank - ndim_ - 1;
                new_config.idist = in_strides[batch_dim];
                new_config.odist = out_strides[batch_dim];
            } else {
                 // Batch=1: Set safe defaults
                if (requested_type == CUFFT_R2C || requested_type == CUFFT_D2Z) {
                    // R2C
                    if (new_config.is_inplace) {
                        // For Batch=1, idist/odist are ignored by cuFFT,
                        // but we calculate them correctly for consistency.
                        int complex_size = total_transform_size / new_config.n.back() * (new_config.n.back()/2 + 1);
                        new_config.idist = complex_size * 2;
                        new_config.odist = complex_size;
                    } else {
                        new_config.idist = total_transform_size;
                        new_config.odist = total_transform_size / new_config.n.back() * (new_config.n.back()/2 + 1);
                    }
                } else {
                    // C2R
                    if (new_config.is_inplace) {
                        int complex_size = total_transform_size / new_config.n.back() * (new_config.n.back()/2 + 1);
                        new_config.idist = complex_size;
                        new_config.odist = complex_size * 2;
                    } else {
                        new_config.idist = total_transform_size / new_config.n.back() * (new_config.n.back()/2 + 1);
                        new_config.odist = total_transform_size;
                    }
                }
            }

        } else if (ndim_ == 1 && axes_.size() == 1) {
            int ax = axes_[0];
            new_config.batch = 1;
            for(int i=0; i<full_rank; ++i) if(i != ax) new_config.batch *= shape_[i];

            new_config.istride = in_strides[ax];
            new_config.ostride = out_strides[ax];

            int dist_axis = -1;
            for (int i = full_rank - 1; i >= 0; --i) {
                if (i != ax) { dist_axis = i; break; }
            }
            if (dist_axis >= 0) {
                new_config.idist = in_strides[dist_axis];
                new_config.odist = out_strides[dist_axis];
            }
            new_config.inembed = new_config.n;
            new_config.onembed = new_config.n;
        } else {
            // Fallback
            new_config.batch = 1;
            new_config.istride = 1; new_config.idist = total_transform_size;
            new_config.ostride = 1; new_config.odist = total_transform_size;
            new_config.inembed = new_config.n;
            new_config.onembed = new_config.n;
        }

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
                                      ndim_, new_config.n.data(),
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

        switch (current_config_.type) {
            case CUFFT_Z2Z: CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD)); break;
            case CUFFT_C2C: CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD)); break;
            case CUFFT_D2Z: CUFFT_CHECK(cufftExecD2Z(plan_, (cufftDoubleReal*)i_ptr, (cufftDoubleComplex*)o_ptr)); break;
            case CUFFT_R2C: CUFFT_CHECK(cufftExecR2C(plan_, (cufftReal*)i_ptr, (cufftComplex*)o_ptr)); break;
            default: break;
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

        switch (current_config_.type) {
            case CUFFT_Z2Z: CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE)); break;
            case CUFFT_C2C: CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE)); break;
            case CUFFT_Z2D: CUFFT_CHECK(cufftExecZ2D(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleReal*)o_ptr)); break;
            case CUFFT_C2R: CUFFT_CHECK(cufftExecC2R(plan_, (cufftComplex*)i_ptr, (cufftReal*)o_ptr)); break;
            default: break;
        }

        double factor = 1.0 / (double)current_config_.logical_size;
        out.attr("__imul__")(factor);

        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

CUFFT_SERIAL::CUFFT_SERIAL(int ndim, const std::vector<int>& shape,
                           const std::vector<int>& axes, const std::string& dtype)
{
    if (!axes.empty()) {
        impl_ = std::make_unique<CUFFT_Generic>(shape, axes, dtype);
    } else {
        if (dtype == "complex128" || dtype == "complex64") {
            impl_ = std::make_unique<CUFFT_C2C_Impl>(ndim, shape, dtype);
        } else {
            impl_ = std::make_unique<CUFFT_R2C_Impl>(ndim, shape, dtype);
        }
    }
}

void CUFFT_SERIAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void CUFFT_SERIAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
CUFFT_SERIAL::~CUFFT_SERIAL() = default;