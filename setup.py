"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.

anyFFT Build Script

This script builds the C++/CUDA/HIP bindings for anyFFT. It automatically attempts
to detect installed libraries (FFTW, CUDA, ROCm, MPI), but you can override the
detection logic by setting the following environment variables before installation:

General Build:
    MAX_JOBS        : Number of parallel workers for compiling GPU source files.
                      (Defaults to total CPU cores).

CPU Backends:
    FFTW_ROOT       : Path to the FFTW3 installation.
    MPI_HOME        : Path to the MPI installation. If set, bypasses mpi4py detection.

GPU Backends (Local & Distributed):
    CUDA_HOME       : Path to the CUDA Toolkit (defaults to /usr/local/cuda).
    ROCM_HOME       : Path to the AMD ROCm Toolkit (defaults to /opt/rocm).
    USE_HIP         : Set to "1" to force building with HIP/ROCm instead of CUDA on
                      hybrid systems where both compilers are present.
    CUFFTMP_ROOT    : Path to the NVIDIA cuFFTMp library headers/binaries.
    NVSHMEM_HOME    : Path to the NVIDIA NVSHMEM library.
    ROCSHMEM_HOME   : Path to the AMD rocSHMEM library.
"""

import os
import sys
import shutil
import subprocess
import sysconfig
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11
import numpy


_brew_cache = {}

def get_brew_prefix(package):
    if sys.platform != "darwin":
        return None
    if package not in _brew_cache:
        try:
            res = subprocess.run(["brew", "--prefix", package], capture_output=True, text=True, check=True)
            _brew_cache[package] = res.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            _brew_cache[package] = None
    return _brew_cache[package]

def get_env_path(env_var, default=None):
    val = os.environ.get(env_var, default)
    if val:
        p = Path(val)
        if p.is_dir():
            return str(p)
    return None

def find_c_dependency(name, header_name, candidate_paths):
    for base in [_f for _f in candidate_paths if _f]:
        base_p = Path(base)
        for inc_sub in ["include", "math_libs/include"]:
            inc_path = base_p / inc_sub
            if (inc_path / header_name).exists():
                for lib_sub in ["lib64", "lib"]:
                    lib_path = base_p / lib_sub
                    if lib_path.is_dir():
                        print(f"[{name}] Found headers at: {inc_path}")
                        return {"found": True, "include": str(inc_path), "lib": str(lib_path)}

    print(f"[{name}] NOTICE: Not found. Skipping/Missing {header_name}.")
    return {"found": False, "include": "", "lib": ""}


# Thread Control for Compilation
try:
    MAX_JOBS = int(os.environ.get("MAX_JOBS", os.cpu_count() or 1))
    MAX_JOBS = max(1, MAX_JOBS)
except ValueError:
    MAX_JOBS = os.cpu_count() or 1

# MPI Detection
mpi_info = {"enabled": False, "includes": [], "lib_dirs": [], "libs": []}
mpi_home = get_env_path("MPI_HOME")

if mpi_home:
    mpi_p = Path(mpi_home)
    mpi_info.update({
        "enabled": True, "includes": [str(mpi_p / "include")],
        "lib_dirs": [str(mpi_p / "lib")], "libs": ["mpi"]
    })
    print(f"[MPI]     Using explicitly defined MPI_HOME: {mpi_home}")
else:
    try:
        import mpi4py
        mpi_candidates = mpi4py.get_config().get("include_dirs", []) + [
            "/opt/homebrew", "/usr/local", "/usr"
        ]
        res = find_c_dependency("MPI", "mpi.h", mpi_candidates)
        if res["found"]:
            mpi_info.update({
                "enabled": True, "libs": ["mpi"],
                "includes": [mpi4py.get_include(), res["include"]],
                "lib_dirs": [res["lib"]]
            })
    except ImportError:
        print("[MPI]     mpi4py not found. Skipping MPI backends.")

# FFTW Detection
fftw_candidates = [get_env_path("FFTW_ROOT"), get_brew_prefix("fftw"), "/usr/local", "/usr"]
fftw_info = find_c_dependency("FFTW", "fftw3.h", fftw_candidates)

# OpenMP Detection
omp_info = {"found": False, "flags": [], "libs": [], "includes": [], "lib_dirs": []}
if sys.platform == "darwin":
    omp_res = find_c_dependency("OpenMP", "omp.h", [get_brew_prefix("libomp"), "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"])
    if omp_res["found"]:
        omp_info.update({
            "found": True, "includes": [omp_res["include"]], "lib_dirs": [omp_res["lib"]],
            "libs": ["omp"], "flags": ["-Xpreprocessor", "-fopenmp"]
        })
else:
    print("[OpenMP]  Enabling OpenMP (Standard Linux/GCC default).")
    omp_info.update({"found": True, "flags": ["-fopenmp"]})

# CUDA & HIP Detection
cuda_home = get_env_path("CUDA_HOME", "/usr/local/cuda")
rocm_home = get_env_path("ROCM_HOME", "/opt/rocm")

nvcc_path = shutil.which("nvcc")
if not nvcc_path and cuda_home:
    candidate = Path(cuda_home) / "bin" / "nvcc"
    if candidate.exists():
        nvcc_path = str(candidate)

hipcc_path = shutil.which("hipcc")
if not hipcc_path and rocm_home:
    candidate = Path(rocm_home) / "bin" / "hipcc"
    if candidate.exists():
        hipcc_path = str(candidate)

USE_HIP = os.environ.get("USE_HIP", "0") == "1"
has_cuda = (nvcc_path is not None) and not USE_HIP
has_hip = (hipcc_path is not None) and (USE_HIP or not has_cuda)

if not fftw_info["found"] and not has_cuda and not has_hip:
    print("\nCRITICAL ERROR: Neither FFTW, CUDA, nor HIP was found.")
    print("                Please install FFTW, a CUDA Toolkit, or ROCm.")
    sys.exit(1)


sources = ["cpp/src/module.cpp"]
libraries = ["m"]
library_dirs = []
define_macros = []
extra_compile_args = ["-O3", "-Wall", "-std=c++17"] + omp_info["flags"]
extra_link_args = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else omp_info["flags"]

include_dirs = [
    pybind11.get_include(), numpy.get_include(),
    sysconfig.get_path("include"), sysconfig.get_path("platinclude"),
    "cpp/include", "cpp/src"
] + omp_info["includes"]

library_dirs.extend(omp_info["lib_dirs"])

# CPU Backend
if fftw_info["found"]:
    sources.append("cpp/src/cpu/fftw.cpp")
    define_macros.append(("ENABLE_FFTW", None))
    include_dirs.append(fftw_info["include"])
    library_dirs.append(fftw_info["lib"])
    libraries.extend(["fftw3", "fftw3f"])

    if omp_info["found"]:
        libraries.extend(["fftw3_omp", "fftw3f_omp"] + omp_info["libs"])
        define_macros.append(("ENABLE_FFTW_OMP", None))

    if mpi_info["enabled"]:
        sources.append("cpp/src/cpu/fftw_mpi.cpp")
        define_macros.append(("ENABLE_FFTW_MPI", None))
        include_dirs.extend(mpi_info["includes"])
        library_dirs.extend(mpi_info["lib_dirs"])
        libraries.extend(["fftw3_mpi", "fftw3f_mpi"] + mpi_info["libs"])

# GPU Backend
cmdclass = {}

if has_cuda:
    print(f"[CUDA]    Found NVCC at: {nvcc_path}")
    sources.append("cpp/src/gpu/gpufft.cu")
    define_macros.append(("ENABLE_CUDA", None))
    include_dirs.append(str(Path(cuda_home) / "include"))

    cuda_lib64 = Path(cuda_home) / "lib64"
    library_dirs.append(str(cuda_lib64) if cuda_lib64.is_dir() else str(Path(cuda_home) / "lib"))
    libraries.extend(["cufft", "cudart"])

    if mpi_info["enabled"]:
        cufftmp = find_c_dependency("cuFFTMp", "cufftMp.h", [get_env_path("CUFFTMP_ROOT"), cuda_home])
        if cufftmp["found"]:
            sources.extend(["cpp/src/gpu/cufftmp.cu", "cpp/src/gpu/gpufft_dist.cu"])
            define_macros.append(("ENABLE_CUDA_MPI", None))
            include_dirs.extend(mpi_info["includes"] + [cufftmp["include"]])
            library_dirs.extend(mpi_info["lib_dirs"] + [cufftmp["lib"]])
            libraries.extend(["cufftMp"] + mpi_info["libs"])

            nvshmem = find_c_dependency("NVSHMEM", "nvshmem.h", [get_env_path("NVSHMEM_HOME"), cuda_home, "/usr/local/nvshmem", "/usr"])
            if nvshmem["found"]:
                include_dirs.append(nvshmem["include"])
                library_dirs.append(nvshmem["lib"])
                libraries.append("nvshmem")
                define_macros.append(("USE_NVSHMEM", None))

elif has_hip:
    print(f"[HIP]     Found HIPCC at: {hipcc_path}")
    sources.append("cpp/src/gpu/gpufft.cu")
    define_macros.append(("__HIP_PLATFORM_AMD__", None))
    define_macros.append(("ENABLE_HIP", None))

    rocm_inc = Path(rocm_home) / "include"
    include_dirs.extend([
        str(rocm_inc),
        str(rocm_inc / "hipfft"),
        str(rocm_inc / "rocfft")
    ])

    rocm_lib64 = Path(rocm_home) / "lib64"
    library_dirs.append(str(rocm_lib64) if rocm_lib64.is_dir() else str(Path(rocm_home) / "lib"))

    libraries.extend(["hipfft", "rocfft", "amdhip64"])

    if mpi_info["enabled"]:
        print("[HIP]     Enabling GPU MPI support (Unified Backend).")
        sources.append("cpp/src/gpu/gpufft_dist.cu")
        define_macros.append(("ENABLE_HIP_MPI", None))
        include_dirs.extend(mpi_info["includes"])
        library_dirs.extend(mpi_info["lib_dirs"])
        libraries.extend(mpi_info["libs"])

        rocshmem = find_c_dependency("rocSHMEM", "rocshmem/rocshmem.hpp", [get_env_path("ROCSHMEM_HOME"), rocm_home, "/opt/rocm", "/usr"])
        if rocshmem["found"]:
            include_dirs.append(rocshmem["include"])
            library_dirs.append(rocshmem["lib"])
            libraries.append("rocshmem")
            define_macros.append(("USE_ROCSHMEM", None))


if has_cuda or has_hip:
    class BuildExtensionGPU(build_ext):
        def build_extensions(self):
            for ext in self.extensions:
                gpu_sources = [s for s in ext.sources if s.endswith(".cu")]
                ext.sources = [s for s in ext.sources if not s.endswith(".cu")]

                def compile_gpu_source(source):
                    src_path = Path(source)
                    obj_path = Path(self.build_temp) / src_path.with_suffix(".o")

                    # Ensure the destination directory exists
                    obj_path.parent.mkdir(parents=True, exist_ok=True)

                    if has_cuda:
                        cmd = [nvcc_path, "-c", str(src_path), "-o", str(obj_path), "-std=c++17", "-Xcompiler", "-fPIC", "-arch=native"]
                    else:
                        cmd = [hipcc_path, "-c", str(src_path), "-o", str(obj_path), "-std=c++17", "-fPIC"]

                    cmd.extend([f"-D{m[0]}" for m in ext.define_macros])
                    for inc in ext.include_dirs: cmd.extend(["-I", inc])

                    subprocess.check_call(cmd)
                    return str(obj_path)

                if gpu_sources:
                    print(f"[Build]   Compiling {len(gpu_sources)} GPU files using {MAX_JOBS} threads...")
                    with ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
                        obj_files = list(executor.map(compile_gpu_source, gpu_sources))
                    ext.extra_objects.extend(obj_files)

            super().build_extensions()

    cmdclass["build_ext"] = BuildExtensionGPU


setup(
    ext_modules=[Extension(
        "anyFFT._core",
        sources=sources,
        include_dirs=list(dict.fromkeys(include_dirs)),
        library_dirs=list(dict.fromkeys(library_dirs)),
        libraries=list(dict.fromkeys(libraries)),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        language="c++",
    )],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    cmdclass=cmdclass,
)
