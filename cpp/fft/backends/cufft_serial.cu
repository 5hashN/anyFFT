#include "cufft_serial.cuh"

// Helper Macros
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
}
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFT Error: " + std::to_string(err)); \
}

// Normalization Kernel
template<typename T>
__global__ void scale_kernel(T* data, T scale_factor, long long n_elements) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = data[idx] / scale_factor;
    }
}

// C2C
class CUFFT_C2C_Impl : public FFTBase {
    cufftHandle plan_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;

public:
    CUFFT_C2C_Impl(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim)
    {
        for (int s : shape) N_ *= s;
        CUFFT_CHECK(cufftCreate(&plan_));

        cufftType t_c2c = (dtype_ == "complex128") ? CUFFT_Z2Z : CUFFT_C2C;

        std::vector<int> n(shape.begin(), shape.end());
        CUFFT_CHECK(cufftPlanMany(&plan_, ndim_, n.data(),
                                  nullptr, 1, 0,
                                  nullptr, 1, 0,
                                  t_c2c, 1));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD));
        } else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE));
        } else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE));
        }

        // Normalize C2C: Output is complex, so size is 2 * N
        // We cast to real pointer to normalize both parts
        ssize_t total_scalars = N_ * 2;
        int block = 256;
        int grid = (total_scalars + block - 1) / block;

        if (dtype_ == "complex128") {
             scale_kernel<double><<<grid, block>>>((double*)o_ptr, (double)N_, total_scalars);
        } else {
             scale_kernel<float><<<grid, block>>>((float*)o_ptr, (float)N_, total_scalars);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_C2C_Impl() {
        cufftDestroy(plan_);
    }
};

// R2C
class CUFFT_R2C_Impl : public FFTBase {
    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;
    ssize_t N_;
    std::string dtype_;
    int ndim_;

public:
    CUFFT_R2C_Impl(int ndim, const std::vector<int>& shape, std::string dtype)
        : N_(1), dtype_(dtype), ndim_(ndim)
    {
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
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)r_ptr, (cufftDoubleComplex*)c_ptr));
        } else {
            CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)r_ptr, (cufftComplex*)c_ptr));
        }
    }

    void backward(py::object in, py::object out) override {
        uintptr_t c_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t r_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "float64") {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)c_ptr, (cufftDoubleReal*)r_ptr));
        } else {
            CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)c_ptr, (cufftReal*)r_ptr));
        }

        // Determine physical size of the output array.
        // If Out-of-Place: out.size() == N (Logical Size).
        // If In-Place: out.size() == Complex_Size (Padded Size).
        // The py::array object 'out' knows its true size.
        ssize_t total_elements = out.attr("size").cast<ssize_t>();

        // If 'out' is a complex array (In-Place case), size() reports complex elements.
        // We need the number of real scalars (floats/doubles), so multiply by 2.
        char kind = out.attr("dtype").attr("kind").cast<char>();
        if (kind == 'c') {
            total_elements *= 2;
        }

        int block = 256;
        int grid = (total_elements + block - 1) / block;

        if (dtype_ == "float64")
            scale_kernel<double><<<grid, block>>>((double*)r_ptr, (double)N_, total_elements);
        else
            scale_kernel<float><<<grid, block>>>((float*)r_ptr, (float)N_, total_elements);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_R2C_Impl() {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
};

// Wrapper Constructor
CUFFT_SERIAL::CUFFT_SERIAL(int ndim, const std::vector<int>& shape, const std::string& dtype)
{
    // Check for C2C
    if (dtype == "complex128" || dtype == "complex64") {
        impl_ = std::make_unique<CUFFT_C2C_Impl>(ndim, shape, dtype);
    }
    // Default to R2C
    else {
        impl_ = std::make_unique<CUFFT_R2C_Impl>(ndim, shape, dtype);
    }
}

void CUFFT_SERIAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void CUFFT_SERIAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
CUFFT_SERIAL::~CUFFT_SERIAL() = default;