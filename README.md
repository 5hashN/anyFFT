# anyFFT

![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Python-blue)
![Backend](https://img.shields.io/badge/backends-FFTW3%20%7C%20CUDA%20%7C%20MPI-green)

anyFFT is a high-performance Python wrapper for standard FFT libraries, providing a unified interface for Serial and Parallel (MPI) transforms on both CPUs and GPUs. It avoids Python overhead by using `pybind11` for direct memory access and supports distributed memory (MPI) out of the box to expose C++ speed with Python convenience.

```Plain text
[Python Layer]  User Interface (NumPy/CuPy)
      │
      │
[C++ Layer]     pybind11 Bindings (Zero-Copy)
      │
      │
[Dispatcher]
      │
      ├──────── CPU: FFTW3 (Serial / MPI)
      └──────── GPU: cuFFT (Serial / MPI)
```

## Features

- Unified API: Switch between backends (CPU/GPU) without changing your math logic.
- Zero-Copy Overhead: Direct access to underlying pointers prevents expensive data duplication.
- Hardware Support:
  - CPU: FFTW3 (Serial), FFTW3-MPI (Distributed)
  - GPU: NVIDIA cuFFT (Serial), NVIDIA cuFFTMp (Multi-GPU) [Experimental]
- Auto-Detection: The build system automatically detects available libraries and compiles only what your system supports.

## Backend Support (Serial)

anyFFT provides a unified `FFT` class, but the underlying libraries have different hardware constraints.

| Feature | CPU Backend (`fftw`) | GPU Backend (`cufft`) |
| :--- | :--- | :--- |
| **Library** | FFTW3 | NVIDIA cuFFT |
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
  - CUDA Toolkit: For GPU support (`nvcc`, `libcufft`)
  - MPI: OpenMPI or MPICH (required for distributed backends)

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

### Serial FFTW

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
fft = FFT(ndim=2, shape=shape, dtype="float64", backend="fftw",
          input=in_data, output=out_data)

# Execute
fft.forward(in_data, out_data)
```

### Serial cuFFT

```Python
import cupy as cp
from anyFFT import FFT

shape = (128, 128)
# Data must be on GPU (CuPy)
data = cp.random.rand(*shape).astype(cp.float64)
out = cp.zeros((128, 65), dtype=cp.complex128)

# Initialize cuFFT plan
fft = FFT(ndim=2, shape=shape, dtype="float64", backend="cufft")

# Execute on GPU
fft.forward(data, out)
```

### Performance Tuning (Experimental)

You can optimize the CPU backend by enabling multithreading and adjusting the FFTW planner rigor.

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
fft = FFT(ndim=3, shape=(128, 128, 128), dtype="complex128", backend="fftw",
          input=in_data, output=out_data,
          n_threads=8, flags=FFTW_MEASURE)
```

For complete working examples of how to use anyFFT in both serial and parallel configurations, please refer to the test scripts included in the repository.

## License

**Copyright (C) 2026 5hashN. All Rights Reserved.**

This repository is strictly for **demonstration purposes only**.
No license is granted to use, modify, or distribute this code.

**Note on Third-Party Libraries:**
This project contains code interfaces for [FFTW3](http://www.fftw.org/) (GPL) and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).
This repository contains only the wrapper source code and does not include any third-party binaries.
