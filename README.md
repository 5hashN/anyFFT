# anyFFT

anyFFT is a high-performance Python wrapper for standard FFT libraries, providing a unified interface for Serial and Parallel (MPI) transforms on both CPUs and GPUs. It uses `pybind11` to expose C++ speed with Python convenience.

## Features

- Unified API: Switch between backends (CPU/GPU) without changing your math logic.
- CPU Backends:
  - Serial: FFTW3
  - Distributed: FFTW3-MPI (MPI-parallelized 3D FFTs)
- GPU Backends:
  - Serial: NVIDIA cuFFT
  - Distributed: NVIDIA cuFFTMp (Multi-GPU/Multi-Node) [Experimental]
- Auto-Detection: The build system automatically detects available libraries and compiles only what your system supports.

## Prerequisites

- Compiler: C++17 compatible compiler (GCC, Clang, or MSVC).
- Python: 3.8+
- Libraries (Optional but recommended):
  - FFTW3: For CPU support (`libfftw3-dev`, `libfftw3-mpi-dev`)
  - CUDA Toolkit: For GPU support (`nvcc`, `libcufft`)
  - MPI: OpenMPI or MPICH (required for parallel backends)

## Installation

### Standard Install

If your libraries are in standard locations (`/usr` or `/usr/local`):

```Bash
pip install .
```

### Building on Clusters (Custom Paths)

On HPC clusters, libraries often reside in non-standard paths. anyFFT allows you to specify these locations via environment variables before building.

Available Variables:

- `MPI_HOME`: Path to MPI installation (containing `include/mpi.h`).
- `CUDA_HOME`: Path to CUDA Toolkit (default `/usr/local/cuda`).
- `FFTW_ROOT`: Path to FFTW installation.
- `CUFFTMP_ROOT`: Path to NVIDIA HPC SDK Math Libs (for multi-GPU).

Example Build Command:

```Bash
# Example for a cluster with non-standard paths
export MPI_HOME=/opt/openmpi_cuda_aware_4
export CUDA_HOME=/usr/local/cuda
export CUFFTMP_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/math_libs

pip install -v .
```

## Usage

For complete working examples of how to use anyFFT in both serial and parallel configurations, please refer to the test scripts included in the repository.

These scripts demonstrate exactly how to initialize the library, handle memory allocation for different backends (NumPy vs CuPy), and manage MPI data decomposition.

- Serial CPU (FFTW): `test_fftw_serial.py`
- Serial GPU (cuFFT): `test_cufft_serial.py`
- Parallel CPU (FFTW-MPI): `test_fftw_mpi.py`
- Parallel GPU (cuFFTMp): `test_cufft_mpi.py`
