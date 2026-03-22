/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#include "anyfft/gpu/cufftmp.cuh"

// Helper Macros
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err))); \
}
#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) throw std::runtime_error("cuFFTMp Error: " + std::to_string(err)); \
}

static bool shmem_initialized = false;

inline void init_shmem_if_needed(MPI_Comm comm) {
#ifdef USE_NVSHMEM
    if (!shmem_initialized) {
        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &comm;

        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        shmem_initialized = true;
    }
#endif
}

inline void* allocate_workspace(size_t size) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
#ifdef USE_NVSHMEM
    ptr = nvshmem_malloc(size);
    if (!ptr) throw std::runtime_error("nvshmem_malloc failed to allocate workspace.");
#else
    CUDA_CHECK(cudaMalloc(&ptr, size));
#endif
    return ptr;
}

inline void free_workspace(void* ptr) {
    if (!ptr) return;
#ifdef USE_NVSHMEM
    nvshmem_free(ptr);
#else
    cudaFree(ptr);
#endif
}

void setup_local_device(MPI_Comm comm) {
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);

    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % num_devices));
    }

    MPI_Comm_free(&node_comm);
}

template<typename T>
__global__ void scale_kernel_mpi(T* data, T scale_factor, long long n_elements) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] = data[idx] / scale_factor;
    }
}

struct Box3D {
    long long lower[3];
    long long upper[3];
};

// Helper: Calculate Local Box based on Rank and Process Grid [p0, p1]
Box3D calculate_local_box(int rank, const std::vector<int>& proc_grid,
                          const std::vector<int>& global_shape,
                          bool is_r2c_complex_side)
{
    int ndim = global_shape.size();
    Box3D box = {{0,0,0}, {0,0,0}};

    // Determine Full Global Shape for this side (Real vs Complex)
    std::vector<long long> shape(3, 1);
    for(int i=0; i<ndim; ++i) shape[i] = global_shape[i];

    if (is_r2c_complex_side) {
        // Complex side of R2C has (N/2 + 1) in the last dimension
        shape[ndim-1] = shape[ndim-1] / 2 + 1;
    }

    // Set defaults (Full Box covers everything)
    for(int i=0; i<3; ++i) box.upper[i] = shape[i];

    // Parse Process Grid
    // [p0, p1]. If p1=1 (or not provided), it is 1D Slab.
    // If p0>1 and p1>1, it is 2D Pencil.
    int p0 = proc_grid.empty() ? 1 : proc_grid[0];
    int p1 = proc_grid.size() > 1 ? proc_grid[1] : 1;

    // Map linear rank to (row, col) coordinates in the process grid
    int r_row = rank / p1;
    int r_col = rank % p1;

    // Calculate Split for Dim 0 (Slowest dimension)
    if (p0 > 1) {
        long long N0 = shape[0];
        long long base0 = N0 / p0;
        long long rem0 = N0 % p0;
        long long start0 = r_row * base0 + (r_row < rem0 ? r_row : rem0);
        long long size0 = base0 + (r_row < rem0 ? 1 : 0);

        box.lower[0] = start0;
        box.upper[0] = start0 + size0;
    }

    // Calculate Split for Dim 1 (Next slowest dimension)
    // Only applies if we have a 2nd dimension to split (ndim >= 2)
    if (p1 > 1) {
        if (ndim < 2) throw std::runtime_error("cuFFT MPI: Cannot use 2D Pencil decomposition on 1D Data.");

        long long N1 = shape[1];
        long long base1 = N1 / p1;
        long long rem1 = N1 % p1;
        long long start1 = r_col * base1 + (r_col < rem1 ? r_col : rem1);
        long long size1 = base1 + (r_col < rem1 ? 1 : 0);

        box.lower[1] = start1;
        box.upper[1] = start1 + size1;
    }

    return box;
}

// Returns: (local_shape_in, local_start_in, local_shape_out, local_start_out)
std::tuple<std::vector<long>, std::vector<long>, std::vector<long>, std::vector<long>>
CUFFT_MPI::get_local_info(int ndim, const std::vector<int>& global_shape,
                          const std::vector<int>& proc_grid, int comm_handle, bool r2c)
{
    MPI_Comm comm = get_mpi_comm(comm_handle);
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Calculate Real Box (Input)
    Box3D real_box = calculate_local_box(rank, proc_grid, global_shape, false);

    std::vector<long> shape_in, start_in;
    for(int i=0; i<ndim; ++i) {
        shape_in.push_back(real_box.upper[i] - real_box.lower[i]);
        start_in.push_back(real_box.lower[i]);
    }

    if (!r2c) {
        // C2C: Input and Output layouts are identical
        return {shape_in, start_in, shape_in, start_in};
    } else {
        // R2C: Output is Complex Box
        Box3D complex_box = calculate_local_box(rank, proc_grid, global_shape, true);
        std::vector<long> shape_out, start_out;
        for(int i=0; i<ndim; ++i) {
            shape_out.push_back(complex_box.upper[i] - complex_box.lower[i]);
            start_out.push_back(complex_box.lower[i]);
        }
        return {shape_in, start_in, shape_out, start_out};
    }
}

void apply_distribution(cufftHandle plan, int ndim, const std::vector<int>& global_shape,
                        const std::vector<int>& proc_grid, int comm_handle, bool is_r2c)
{
    MPI_Comm comm = get_mpi_comm(comm_handle);
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Calculate Input Box
    Box3D box_in = calculate_local_box(rank, proc_grid, global_shape, false);

    // Calculate Output Box
    // For R2C: Output is Complex (Halved last dim)
    // For C2C: Output is same as Input
    Box3D box_out;
    if (is_r2c) box_out = calculate_local_box(rank, proc_grid, global_shape, true);
    else box_out = box_in;

    // Convert to arrays for cufftXtSetDistribution
    long long input_lower[3] = {0,0,0}, input_upper[3] = {1,1,1}, input_strides[3] = {1,1,1};
    long long output_lower[3] = {0,0,0}, output_upper[3] = {1,1,1}, output_strides[3] = {1,1,1};

    // Fill strictly {dim0, dim1, dim2} order
    for(int i=0; i<ndim; ++i) {
        input_lower[i] = box_in.lower[i];
        input_upper[i] = box_in.upper[i]; // Upper bound is exclusive (start + size)

        output_lower[i] = box_out.lower[i];
        output_upper[i] = box_out.upper[i];
    }

    CUFFT_CHECK(cufftXtSetDistribution(plan, ndim,
                                       input_lower, input_upper,
                                       output_lower, output_upper,
                                       input_strides, output_strides));
}

class CUFFT_DIST_C2C : public FFTBase {
    cufftHandle plan_;
    void* work_area_ = nullptr;
    ssize_t global_N_;
    std::string dtype_;
    MPI_Comm comm_;

public:
    CUFFT_DIST_C2C(int ndim, const std::vector<int>& shape, const std::vector<int>& proc_grid,
                   int comm_handle, std::string dtype)
        : global_N_(1), dtype_(dtype)
    {
        comm_ = get_mpi_comm(comm_handle);
        setup_local_device(comm_);
        init_shmem_if_needed(comm_);

        for (int s : shape) global_N_ *= s;

        CUFFT_CHECK(cufftCreate(&plan_));
        CUFFT_CHECK(cufftMpAttachComm(plan_, CUFFT_COMM_MPI, &comm_));

        // Apply Slab/Pencil Distribution
        apply_distribution(plan_, ndim, shape, proc_grid, comm_handle, false);

        size_t ws = 0;
        cufftType t_c2c = (dtype_ == "complex128") ? CUFFT_Z2Z : CUFFT_C2C;
        std::vector<long long> n_long(shape.begin(), shape.end());

        if (ndim == 3) CUFFT_CHECK(cufftMakePlan3d(plan_, n_long[0], n_long[1], n_long[2], t_c2c, &ws));
        else CUFFT_CHECK(cufftMakePlan2d(plan_, n_long[0], n_long[1], t_c2c, &ws));

        work_area_ = allocate_workspace(ws);
        if (ws > 0) CUFFT_CHECK(cufftSetWorkArea(plan_, work_area_));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        py::gil_scoped_release release;
        if (dtype_ == "complex128")
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_FORWARD))
        else
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_FORWARD));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();
        ssize_t local_elements = out.attr("size").cast<ssize_t>();

        // Adjust elements count for complex types if numpy reports strictly elements
        char kind = out.attr("dtype").attr("kind").cast<char>();
        if (kind == 'c') local_elements *= 2;

        int block = 256;
        int grid = (local_elements + block - 1) / block;

        py::gil_scoped_release release;
        if (dtype_ == "complex128")
            CUFFT_CHECK(cufftExecZ2Z(plan_, (cufftDoubleComplex*)i_ptr, (cufftDoubleComplex*)o_ptr, CUFFT_INVERSE))
        else
            CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)i_ptr, (cufftComplex*)o_ptr, CUFFT_INVERSE));

        if (dtype_ == "complex128")
            scale_kernel_mpi<double><<<grid, block>>>((double*)o_ptr, (double)global_N_, local_elements);
        else
            scale_kernel_mpi<float><<<grid, block>>>((float*)o_ptr, (float)global_N_, local_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_DIST_C2C() {
        free_workspace(work_area_);
        cufftDestroy(plan_);
    }
};

class CUFFT_DIST_R2C : public FFTBase {
    cufftHandle plan_r2c_;
    cufftHandle plan_c2r_;
    void* work_area_r2c_ = nullptr;
    void* work_area_c2r_ = nullptr;

    ssize_t global_N_;
    std::string dtype_;
    bool inplace_;

public:
    CUFFT_DIST_R2C(int ndim, const std::vector<int>& shape, const std::vector<int>& proc_grid,
                   int comm_handle, std::string dtype, bool inplace)
        : global_N_(1), dtype_(dtype), inplace_(inplace)
    {
        MPI_Comm comm = get_mpi_comm(comm_handle);
        setup_local_device(comm);
        init_shmem_if_needed(comm);

        for (int s : shape) global_N_ *= s;

        CUFFT_CHECK(cufftCreate(&plan_r2c_));
        CUFFT_CHECK(cufftCreate(&plan_c2r_));
        CUFFT_CHECK(cufftMpAttachComm(plan_r2c_, CUFFT_COMM_MPI, &comm));
        CUFFT_CHECK(cufftMpAttachComm(plan_c2r_, CUFFT_COMM_MPI, &comm));

        // DISTRIBUTIONS
        int rank;
        MPI_Comm_rank(comm, &rank);

        // Calculate logical boxes (Valid Data)
        Box3D real_box = calculate_local_box(rank, proc_grid, shape, false);
        Box3D complex_box = calculate_local_box(rank, proc_grid, shape, true);

        long long real_lower[3]={0}, real_upper[3]={1}, real_strides[3]={1};
        long long comp_lower[3]={0}, comp_upper[3]={1}, comp_strides[3]={1};

        for(int i=0; i<ndim; ++i) {
            real_lower[i] = real_box.lower[i]; real_upper[i] = real_box.upper[i];
            comp_lower[i] = complex_box.lower[i]; comp_upper[i] = complex_box.upper[i];
        }

        // Apply to R2C Plan: Input=Real, Output=Complex
        CUFFT_CHECK(cufftXtSetDistribution(plan_r2c_, ndim,
                                          real_lower, real_upper,
                                          comp_lower, comp_upper,
                                          real_strides, comp_strides));

        // Apply to C2R Plan: Input=Complex, Output=Real
        CUFFT_CHECK(cufftXtSetDistribution(plan_c2r_, ndim,
                                          comp_lower, comp_upper,
                                          real_lower, real_upper,
                                          comp_strides, real_strides));

        size_t ws_r2c=0, ws_c2r=0;
        cufftType t_r2c = (dtype_ == "float64") ? CUFFT_D2Z : CUFFT_R2C;
        cufftType t_c2r = (dtype_ == "float64") ? CUFFT_Z2D : CUFFT_C2R;
        std::vector<long long> n_long(shape.begin(), shape.end());

        if (ndim == 3) {
            CUFFT_CHECK(cufftMakePlan3d(plan_r2c_, n_long[0], n_long[1], n_long[2], t_r2c, &ws_r2c));
            CUFFT_CHECK(cufftMakePlan3d(plan_c2r_, n_long[0], n_long[1], n_long[2], t_c2r, &ws_c2r));
        } else {
            CUFFT_CHECK(cufftMakePlan2d(plan_r2c_, n_long[0], n_long[1], t_r2c, &ws_r2c));
            CUFFT_CHECK(cufftMakePlan2d(plan_c2r_, n_long[0], n_long[1], t_c2r, &ws_c2r));
        }

        work_area_r2c_ = allocate_workspace(ws_r2c);
        if (ws_r2c > 0) CUFFT_CHECK(cufftSetWorkArea(plan_r2c_, work_area_r2c_));

        work_area_c2r_ = allocate_workspace(ws_c2r);
        if (ws_c2r > 0) CUFFT_CHECK(cufftSetWorkArea(plan_c2r_, work_area_c2r_));
    }

    void forward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        if (inplace_ && i_ptr != o_ptr) throw std::runtime_error("cuFFT MPI R2C: R2C configured as In-Place, but pointers differ.");

        py::gil_scoped_release release;
        if (dtype_ == "float64")
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_, (cufftDoubleReal*)i_ptr, (cufftDoubleComplex*)o_ptr))
        else
            CUFFT_CHECK(cufftExecR2C(plan_r2c_, (cufftReal*)i_ptr, (cufftComplex*)o_ptr));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backward(py::object in, py::object out) override {
        uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
        uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();

        py::gil_scoped_release release;
        if (dtype_ == "float64")
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_, (cufftDoubleComplex*)i_ptr, (cufftDoubleReal*)o_ptr))
        else
            CUFFT_CHECK(cufftExecC2R(plan_c2r_, (cufftComplex*)i_ptr, (cufftReal*)o_ptr));

        // Normalize
        ssize_t out_elements = out.attr("size").cast<ssize_t>();
        int block = 256;
        int grid = (out_elements + block - 1) / block;

        if (dtype_ == "float64")
            scale_kernel_mpi<double><<<grid, block>>>((double*)o_ptr, (double)global_N_, out_elements);
        else
            scale_kernel_mpi<float><<<grid, block>>>((float*)o_ptr, (float)global_N_, out_elements);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~CUFFT_DIST_R2C() {
        free_workspace(work_area_r2c_);
        free_workspace(work_area_c2r_);
        cufftDestroy(plan_r2c_);
        cufftDestroy(plan_c2r_);
    }
};

CUFFT_DIST::CUFFT_DIST(const std::vector<int>& shape,
                       const std::vector<int>& proc_grid,
                       py::object in, py::object out,
                       int comm_handle,
                       const std::string& dtype)
{
    uintptr_t i_ptr = in.attr("data").attr("ptr").cast<uintptr_t>();
    uintptr_t o_ptr = out.attr("data").attr("ptr").cast<uintptr_t>();
    int ndim = shape.size();
    bool is_inplace = (i_ptr == o_ptr);

    if (dtype == "complex128" || dtype == "complex64")
        impl_ = std::make_unique<CUFFT_DIST_C2C>(ndim, shape, proc_grid, comm_handle, dtype);
    else
        impl_ = std::make_unique<CUFFT_DIST_R2C>(ndim, shape, proc_grid, comm_handle, dtype, is_inplace);
}

void CUFFT_DIST::forward(py::object in, py::object out) { impl_->forward(in, out); }
void CUFFT_DIST::backward(py::object in, py::object out) { impl_->backward(in, out); }
CUFFT_DIST::~CUFFT_DIST() = default;
