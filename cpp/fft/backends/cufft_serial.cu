#include "cufft_serial.cuh"

// Macros for error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
}
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFT Error: " + std::to_string(err)); \
}

// Kernel for normalization
template<typename T>
__global__ void scale_kernel(T* data, T scale_factor, long long n_elements) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = data[idx] / scale_factor;
    }
}

CUFFT_SERIAL::CUFFT_SERIAL(int ndim, const std::vector<int>& shape, const std::string& dtype)
    : shape_(shape), ndim_(ndim), plan_r2c_(0), plan_c2r_(0), N_(1), dtype_(dtype)
{
    for (int s : shape_) N_ *= s;
    CUFFT_CHECK(cufftCreate(&plan_r2c_));
    CUFFT_CHECK(cufftCreate(&plan_c2r_));

    // Determine types
    cufftType t_r2c, t_c2r;
    if (dtype_ == "float64") { t_r2c = CUFFT_D2Z; t_c2r = CUFFT_Z2D; }
    else { t_r2c = CUFFT_R2C; t_c2r = CUFFT_C2R; }

    std::vector<int> n(shape_.begin(), shape_.end());

    // Create plans
    CUFFT_CHECK(cufftPlanMany(&plan_r2c_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_r2c, 1));
    CUFFT_CHECK(cufftPlanMany(&plan_c2r_, ndim_, n.data(), nullptr, 1, 0, nullptr, 1, 0, t_c2r, 1));
}

// Extract pointers from Python objects i.e. CuPy arrays
void CUFFT_SERIAL::forward(py::object real_in, py::object complex_out) {
    uintptr_t r_ptr = real_in.attr("data").attr("ptr").cast<uintptr_t>();
    uintptr_t c_ptr = complex_out.attr("data").attr("ptr").cast<uintptr_t>();

    if (dtype_ == "float64") {
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)r_ptr, (cufftDoubleComplex*)c_ptr));
    } else {
        CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)r_ptr, (cufftComplex*)c_ptr));
    }
}

void CUFFT_SERIAL::backward(py::object complex_in, py::object real_out) {
    uintptr_t c_ptr = complex_in.attr("data").attr("ptr").cast<uintptr_t>();
    uintptr_t r_ptr = real_out.attr("data").attr("ptr").cast<uintptr_t>();

    if (dtype_ == "float64") {
        CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)c_ptr, (cufftDoubleReal*)r_ptr));
    } else {
        CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)c_ptr, (cufftReal*)r_ptr));
    }

    // Normalization
    int block = 256;
    int grid = (N_ + block - 1) / block;
    if (dtype_ == "float64")
        scale_kernel<double><<<grid, block>>>((double*)r_ptr, (double)N_, N_);
    else
        scale_kernel<float><<<grid, block>>>((float*)r_ptr, (float)N_, N_);

    CUDA_CHECK(cudaDeviceSynchronize());
}

CUFFT_SERIAL::~CUFFT_SERIAL() {
    cufftDestroy(plan_r2c_);
    cufftDestroy(plan_c2r_);
}