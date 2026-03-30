# anyFFT

![CI](https://github.com/5hashN/anyFFT/actions/workflows/ci.yml/badge.svg)
![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20CUDA%20%7C%20Python-blue)
![Backend](https://img.shields.io/badge/backends-FFTW3%20%7C%20CuFFT%20%7C%20MPI-green)

anyFFT is a high-performance Python wrapper for standard FFT libraries, providing a unified interface for Local and Distributed (MPI) transforms on both CPUs and GPUs. It avoids Python overhead by using `pybind11` for direct memory access and supports distributed memory (MPI) out of the box to expose C++ speed with Python convenience.

```Plain text
[Python Layer]  User Interface (NumPy/CuPy)
      │
      │
[C++ Layer]     pybind11 Bindings (Zero-Copy)
      │
      │
[Dispatcher]
      │
      ├──────── CPU: FFTW3 / FFTW3-MPI
      └──────── GPU: cuFFT / hipFFT / cuFFTMp
```

## Features

- Unified API: Switch between backends (CPU/GPU) without changing your math logic.
- Zero-Copy Overhead: Direct access to underlying pointers prevents expensive data duplication.
- Hardware Support:
  - CPU: FFTW3 (Local), FFTW3-MPI (Distributed)
  - GPU (NVIDIA): cuFFT (Local), cuFFTMp (Distributed via NVSHMEM)
  - GPU (AMD): hipFFT (Local and Distributed via rocSHMEM) [Experimental]
- Auto-Detection: The build system automatically detects available libraries and compiles only what your system supports.

## Backend Support (Local)

anyFFT provides a unified `FFT` class, but the underlying libraries have different hardware constraints.

| Feature | CPU Backend (`fftw`) | GPU Backend (`cufft`, `hipfft`) |
| :--- | :--- | :--- |
| **Library** | FFTW3 | NVIDIA cuFFT / AMD hipFFT |
| **Data Object** | `numpy.ndarray` | `cupy.ndarray` |
| **Dimensions** | **N-Dimensional** (1D, 2D, 3D, 4D, ...) | **Max 3D** (1D, 2D, 3D) |
| **Transforms** | C2C, R2C, C2R | C2C, R2C, C2R |
| **Placement** | In-Place & Out-of-Place | In-Place & Out-of-Place |
| **Axes** | Arbitrary (e.g., `axes=(0, 2)`) | Arbitrary (e.g., `axes=(0, 2)`) |

> **Note on R2C In-Place Transforms:**
> Both backends require padding for In-Place Real-to-Complex transforms. The last dimension of the input array must be padded to accommodate the complex result (size $N/2 + 1$ complex elements). anyFFT handles the stride calculations automatically, provided the input buffer is sized correctly.

## Prerequisites

- Compiler: C++17 compatible compiler (GCC, Clang, or MSVC).
- Python: 3.8+
- Libraries (Optional but recommended):
  - FFTW3: For CPU support (`libfftw3-dev`, `libfftw3-mpi-dev`)
  - OpenMP: For CPU multithreading (Standard on Linux/GCC, requires `libomp` on macOS)
  - CUDA Toolkit: For NVIDIA GPU support (`nvcc`, `libcufft`)
  - ROCm Toolkit: For AMD GPU support (`hipcc`, `hipfft`, `rocfft`)
  - MPI: OpenMPI or MPICH (detected via `MPI_HOME` or fallback to `mpi4py` paths)

## Installation

### Standard Install

If your libraries are in standard locations (`/usr` or `/usr/local`):

```Bash
pip install .
```

### Building on Clusters (Custom Paths)

On HPC clusters, libraries often reside in non-standard paths. anyFFT allows you to specify these locations via environment variables before building to bypass automatic detection.

Available Build Variables:

- `MAX_JOBS`: Number of parallel workers for compiling GPU source files (defaults to total CPU cores).
- `FFTW_ROOT`: Path to FFTW3 installation.
- `MPI_HOME`: Path to MPI installation (bypasses `mpi4py` detection if set).
- `CUDA_HOME`: Path to CUDA Toolkit (default `/usr/local/cuda`).
- `ROCM_HOME`: Path to AMD ROCm Toolkit (defaults to /opt/rocm).
- `USE_HIP`: Set to "1" to force building with HIP/ROCm instead of CUDA on hybrid systems.
- `CUFFTMP_ROOT`: Path to NVIDIA cuFFTMp library headers/binaries.
- `NVSHMEM_HOME`: Path to the NVIDIA NVSHMEM library.
- `ROCSHMEM_HOME`: Path to the AMD rocSHMEM library.

Example Build Command:

```Bash
# Example for a cluster with non-standard paths
export MPI_HOME=/opt/openmpi_rocm_aware_4
export ROCM_HOME=/opt/rocm-5.4.0
export ROCSHMEM_HOME=/opt/rocm/rocshmem
export USE_HIP=1

pip install -v .
```

## Usage

### Local FFTW

```Python
import numpy as np
from anyFFT import FFT

# Prepare Data
shape = (128, 128)
# Create Real input (for R2C transform)
in_data = np.random.rand(*shape).astype(np.float64)
# Output shape for R2C is (N, M/2 + 1)
out_shape = (128, 128 // 2 + 1)
out_data = np.zeros(out_shape, dtype=np.complex128)

# Initialize FFTW plan
# Note: We pass input/output arrays to optimize the plan for these specific memory layouts
fft = FFT(shape=shape, dtype="float64", backend="fftw",
          input=in_data, output=out_data)

# Execute
fft.forward(in_data, out_data)
```

### Local Unified gpuFFT

```Python
import cupy as cp
from anyFFT import FFT

shape = (128, 128)
# Data must be on GPU (CuPy)
data = cp.random.rand(*shape).astype(cp.float64)
out = cp.zeros((128, 65), dtype=cp.complex128)

# Initialize cuFFT plan
fft = FFT(shape=shape, dtype="float64", backend="gpufft")

# Execute on GPU
fft.forward(data, out)
```

### FFTW Performance Tuning

You can optimize the `fftw` backend by enabling multithreading and adjusting the FFTW planner rigor.

```Python
import numpy as np
from anyFFT import FFT, FFTW_MEASURE, FFTW_PATIENT

# ... setup data ...

# Create a highly optimized plan
# flags: Controls planner rigor.
#        FFTW_ESTIMATE (Default) - Fast setup, reasonable speed.
#        FFTW_MEASURE - Slower setup, faster execution. (Overwrites input during plan!)
#        FFTW_PATIENT - Very slow setup, maximum execution speed.
# n_threads: Number of OMP threads to use (requires library compiled with OpenMP).
fft = FFT(shape=(128, 128, 128), dtype="complex128", backend="fftw",
          input=in_data, output=out_data,
          n_threads=8, flags=FFTW_MEASURE)
```

For complete working examples of how to use anyFFT in both local and distributed configurations, please refer to the test scripts included in the repository.

## License

**Copyright (C) 2026 5hashN. All Rights Reserved.**

This repository is strictly for **demonstration purposes only**.
No license is granted to use, modify, or distribute this code.

**Note on Third-Party Libraries:**
This project contains code interfaces for [FFTW3](http://www.fftw.org/) (GPL), [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) and [AMD ROCm](https://rocm.docs.amd.com/).
This repository contains only the wrapper source code and does not include any third-party binaries.
