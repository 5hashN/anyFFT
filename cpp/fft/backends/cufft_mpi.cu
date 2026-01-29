#include "cufft_mpi.cuh"

// Helper Macros
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
}
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFTMp Error: " + std::to_string(err)); \
}

// Normalization Kernel
template<typename T>
__global__ void scale_kernel_mpi(T* data, T scale_factor, long long n_elements) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = data[idx] / scale_factor;
    }
}

// STATIC HELPER: Layout Calculator
std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
CUFFT_MPI::get_local_info(int ndim, const std::vector<int>& global_shape, int comm_handle, bool r2c) {

    if (ndim < 2 || ndim > 3) {
        throw std::runtime_error("Only 2D and 3D transforms supported.");
    }

    // Use shared helper for consistency
    MPI_Comm comm = get_mpi_comm(comm_handle);

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Slab Decomposition (Split Dim 0)
    long N0 = global_shape[0];
    long base_chunk = N0 / size;
    long remainder = N0 % size;

    long my_n0 = base_chunk + (rank < remainder ? 1 : 0);
    long my_start = (rank * base_chunk) + (rank < remainder ? rank : remainder);

    // Base Shape (Real / Logical Input)
    std::vector<long> shape(global_shape.begin(), global_shape.end());
    shape[0] = my_n0;

    std::vector<long> start(ndim, 0);
    start[0] = my_start;

    if (!r2c) {
        // C2C: Input and Output are identical
        return {shape, start, shape, start};
    } else {
        // R2C: Standard Slab
        // Input is the Real shape [loc_n0, ..., N_last]
        // Output is the Complex shape [loc_n0, ..., N_last/2 + 1]
        std::vector<long> out_shape = shape;
        out_shape.back() = global_shape.back() / 2 + 1;

        // Return {Real_Shape, Start, Complex_Shape, Start}
        return {shape, start, out_shape, start};
    }
}

// MPI C2C In-Place and Out-of-Place
class CUFFT_MPI_C2C : public FFTBase {
    cufftHandle plan_;
    ssize_t global_N_;
    std::string dtype_;
    MPI_Comm comm_;
    int rank_;

public:
    CUFFT_MPI_C2C(int ndim, const std::vector<int>& shape,
                  int comm_handle, std::string dtype)
        : global_N_(1), dtype_(dtype)
    {
        comm_ = get_mpi_comm(comm_handle);
        MPI_Comm_rank(comm_, &rank_);

        for (int s : shape) global_N_ *= s;

        // Create & Attach
        CUFFT_CHECK(cufftCreate(&plan_));
        CUFFT_CHECK(cufftMpAttachComm(plan_, CUFFT_COMM_MPI, &comm_));

        // Make Plan
        size_t workspace;
        cufftType t_c2c = (dtype_ == "complex128") ? CUFFT_Z2Z : CUFFT_C2C;
        std::vector<long long> n_long(shape.begin(), shape.end());

        if (ndim == 3) {
            CUFFT_CHECK(cufftMakePlan3d(plan_, n_long[0], n_long[1], n_long[2], t_c2c, &workspace));
        } else {
            CUFFT_CHECK(cufftMakePlan2d(plan_, n_long[0], n_long[1], t_c2c, &workspace));
        }
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD));
        } else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (dtype_ == "complex128") {
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE));
        } else {
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE));
        }

        // Normalize
        ssize_t local_elements = out.attr("size").cast<ssize_t>();
        char kind = out.attr("dtype").attr("kind").cast<char>();
        if (kind == 'c') local_elements *= 2;

        int block = 256;
        int grid = (local_elements + block - 1) / block;

        if (dtype_ == "complex128") {
            scale_kernel_mpi<double><<<grid, block>>>((double*)o_ptr, (double)global_N_, local_elements);
        } else {
            scale_kernel_mpi<float><<<grid, block>>>((float*)o_ptr, (float)global_N_, local_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_MPI_C2C() {
        cufftDestroy(plan_);
    }
};

// MPI R2C In-Place Only
class CUFFT_MPI_R2C_InPlace : public FFTBase {
    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;
    ssize_t global_N_;
    std::string dtype_;
    MPI_Comm comm_;
    int rank_;

public:
    CUFFT_MPI_R2C_InPlace(int ndim, const std::vector<int>& shape,
                          int comm_handle, std::string dtype)
        : global_N_(1), dtype_(dtype)
    {
        comm_ = get_mpi_comm(comm_handle);
        MPI_Comm_rank(comm_, &rank_);

        for (int s : shape) global_N_ *= s;

        // Create & Attach
        CUFFT_CHECK(cufftCreate(&plan_r2c_));
        CUFFT_CHECK(cufftCreate(&plan_c2r_));
        CUFFT_CHECK(cufftMpAttachComm(plan_r2c_, CUFFT_COMM_MPI, &comm_));
        CUFFT_CHECK(cufftMpAttachComm(plan_c2r_, CUFFT_COMM_MPI, &comm_));

        size_t ws;
        cufftType t_r2c = (dtype_ == "float64") ? CUFFT_D2Z : CUFFT_R2C;
        cufftType t_c2r = (dtype_ == "float64") ? CUFFT_Z2D : CUFFT_C2R;

        std::vector<long long> n_long(shape.begin(), shape.end());

        // Make Plans
        if (ndim == 3) {
            CUFFT_CHECK(cufftMakePlan3d(plan_r2c_, n_long[0], n_long[1], n_long[2], t_r2c, &ws));
            CUFFT_CHECK(cufftMakePlan3d(plan_c2r_, n_long[0], n_long[1], n_long[2], t_c2r, &ws));
        } else {
            CUFFT_CHECK(cufftMakePlan2d(plan_r2c_, n_long[0], n_long[1], t_r2c, &ws));
            CUFFT_CHECK(cufftMakePlan2d(plan_c2r_, n_long[0], n_long[1], t_c2r, &ws));
        }
    }

    void forward(py::object in, py::object out) override {
        // Check Pointers (Runtime enforcement of In-Place)
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (i_ptr != o_ptr) {
            throw std::runtime_error("R2C forward must be In-Place (input and output must match).");
        }

        if (dtype_ == "float64") {
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)i_ptr, (cufftDoubleComplex*)i_ptr));
        } else {
            CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)i_ptr, (cufftComplex*)i_ptr));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (i_ptr != o_ptr) {
            throw std::runtime_error("R2C backward must be In-Place (input and output must match).");
        }

        if (dtype_ == "float64") {
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)i_ptr, (cufftDoubleReal*)i_ptr));
        } else {
            CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)i_ptr, (cufftReal*)i_ptr));
        }

        // Normalize
        ssize_t complex_elements = in.attr("size").cast<ssize_t>();
        ssize_t real_elements = complex_elements * 2;

        int block = 256;
        int grid = (real_elements + block - 1) / block;

        if (dtype_ == "float64") {
             scale_kernel_mpi<double><<<grid, block>>>((double*)i_ptr, (double)global_N_, real_elements);
        } else {
             scale_kernel_mpi<float><<<grid, block>>>((float*)i_ptr, (float)global_N_, real_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_MPI_R2C_InPlace() {
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
};

// The Wrapper Constructor
CUFFT_MPI::CUFFT_MPI(int ndim, const std::vector<int>& shape,
                     py::object in, py::object out,
                     int comm_handle, const std::string& dtype)
{
    if (dtype == "complex128" || dtype == "complex64") {
        impl_ = std::make_unique<CUFFT_MPI_C2C>(ndim, shape, comm_handle, dtype);
    }
    else {
        // Enforce In-Place check at initialization time (consistent with FFTW MPI)
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (i_ptr != o_ptr) {
            throw std::runtime_error("R2C Out-of-Place not supported. Please use In-Place with a padded buffer.");
        }
        impl_ = std::make_unique<CUFFT_MPI_R2C_InPlace>(ndim, shape, comm_handle, dtype);
    }
}

void CUFFT_MPI::forward(py::object in, py::object out) { impl_->forward(in, out); }
void CUFFT_MPI::backward(py::object in, py::object out) { impl_->backward(in, out); }
CUFFT_MPI::~CUFFT_MPI() = default;