import os
import sys
import shutil
import subprocess
import sysconfig
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11
import numpy

# Helper: Path Discovery
def get_env_path(env_var, default=None):
    """Retrieves a path from env var, checking if it exists."""
    path = os.environ.get(env_var, default)
    if path and os.path.isdir(path):
        return path
    return None

# MPI Detection
ENABLE_MPI = False
mpi_include_dirs = []
mpi_lib_dirs = []
mpi_libraries = []

# Check User Environment Variable first
MPI_HOME = get_env_path("MPI_HOME")

if MPI_HOME:
    ENABLE_MPI = True
    print(f"Using MPI_HOME: {MPI_HOME}")
    mpi_include_dirs.append(os.path.join(MPI_HOME, "include"))
    mpi_lib_dirs.append(os.path.join(MPI_HOME, "lib"))
    mpi_libraries.append("mpi") # Link against libmpi
else:
    # Fallback to mpi4py detection
    try:
        import mpi4py
        ENABLE_MPI = True
        mpi_include_dirs.append(mpi4py.get_include())
        # We assume standard linker paths will find 'mpi' if not specified
        mpi_libraries.append("mpi")
        print("MPI_HOME not found. Falling back to mpi4py detection.")
    except ImportError:
        print("mpi4py not found. Skipping MPI backends.")

# FFTW Detection
FFTW_ROOT = get_env_path("FFTW_ROOT")

# If not explicitly set, try to find it via Homebrew (macOS)
if not FFTW_ROOT and sys.platform == "darwin":
    try:
        # Ask brew where fftw is installed
        res = subprocess.run(["brew", "--prefix", "fftw"], capture_output=True, text=True)
        if res.returncode == 0:
            FFTW_ROOT = res.stdout.strip()
    except FileNotFoundError:
        pass # brew not found

# Fallback defaults
if not FFTW_ROOT:
    FFTW_ROOT = "/usr"

FFTW_INC = os.environ.get("FFTW_INC", os.path.join(FFTW_ROOT, "include"))
FFTW_LIB = os.environ.get("FFTW_LIB", os.path.join(FFTW_ROOT, "lib"))

# Fallback: Check /usr/local if not found in /usr
if not os.path.exists(os.path.join(FFTW_INC, "fftw3.h")) and FFTW_ROOT == "/usr":
    FFTW_INC = "/usr/local/include"
    FFTW_LIB = "/usr/local/lib"

HAS_FFTW = os.path.exists(os.path.join(FFTW_INC, "fftw3.h"))

# CUDA Detection
CUDA_HOME = get_env_path("CUDA_HOME", "/usr/local/cuda")
NVCC_PATH = shutil.which("nvcc")

# If nvcc isn't in PATH, try looking in CUDA_HOME
if NVCC_PATH is None and CUDA_HOME:
    candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
    if os.path.exists(candidate):
        NVCC_PATH = candidate

HAS_CUDA = NVCC_PATH is not None

# Check if cufftMp header exists
HAS_CUFFT_MPI = False
if HAS_CUDA and ENABLE_MPI:
    CUFFTMP_ROOT = get_env_path("CUFFTMP_ROOT", CUDA_HOME)

    # Check common subdirectories for the header
    possible_include_paths = [
        os.path.join(CUFFTMP_ROOT, "include"),
        os.path.join(CUFFTMP_ROOT, "math_libs", "include"), # Common in HPC SDK
    ]

    cufftmp_inc_found = None
    for p in possible_include_paths:
        if os.path.exists(os.path.join(p, "cufftMp.h")):
            cufftmp_inc_found = p
            break

    if cufftmp_inc_found:
        HAS_CUFFT_MPI = True
        print(f"Found cuFFTMp header at {cufftmp_inc_found}")
    else:
        print(f"cuFFTMp header not found in {CUFFTMP_ROOT}. Skipping cuFFTMp backend.")

# Safety Check
if not HAS_FFTW and not HAS_CUDA:
    print("CRITICAL ERROR: Neither FFTW nor CUDA was found.")
    print("Please install FFTW (libfftw3-dev) or a CUDA Toolkit.")
    sys.exit(1)

# Compiler & Linker Flags
# Flags for the C++ Compiler
extra_compile_args = ["-O3", "-Wall", "-std=c++17"]

# Flags for the Linker
extra_link_args = []
if sys.platform == "darwin":
    # These are linker flags, not compiler flags
    extra_link_args.extend(["-undefined", "dynamic_lookup"])

# Base includes
include_dirs = [
    pybind11.get_include(),
    numpy.get_include(),
    sysconfig.get_path("include"),
    sysconfig.get_path("platinclude"),
    "cpp/include",
    "cpp/src",
]

sources = ["cpp/src/module.cpp"]
libraries = ["m"]
library_dirs = []
define_macros = []

# Conditional Backend Inclusion
if HAS_FFTW:
    print(f"FFTW found at {FFTW_INC}. Enabling CPU support.")
    sources.append("cpp/src/fftw/fftw_serial.cpp")
    define_macros.append(("ENABLE_FFTW", None))
    include_dirs.append(FFTW_INC)
    library_dirs.append(FFTW_LIB)
    libraries.extend(["fftw3", "fftw3f"])

    if ENABLE_MPI:
        print("Enabling CPU MPI support (FFTW-MPI).")
        sources.append("cpp/src/fftw/fftw_mpi.cpp")
        define_macros.append(("ENABLE_FFTW_MPI", None))
        include_dirs.extend(mpi_include_dirs)
        library_dirs.extend(mpi_lib_dirs)
        libraries.extend(["fftw3_mpi", "fftw3f_mpi"])
        libraries.extend(mpi_libraries)
else:
    print("FFTW not found. Skipping CPU backend.")

cmdclass = {}
if HAS_CUDA:
    print(f"CUDA found at {NVCC_PATH}. Enabling GPU support.")
    sources.append("cpp/src/cufft/cufft_serial.cu")
    define_macros.append(("ENABLE_CUDA", None))
    include_dirs.append(os.path.join(CUDA_HOME, "include"))

    # Handle lib64 vs lib folder structure
    cuda_lib64 = os.path.join(CUDA_HOME, "lib64")
    if os.path.isdir(cuda_lib64):
        library_dirs.append(cuda_lib64)
    else:
        library_dirs.append(os.path.join(CUDA_HOME, "lib"))

    libraries.extend(["cufft", "cudart"])

    if HAS_CUFFT_MPI:
        sources.append("cpp/src/cufft/cufft_mpi.cu")
        define_macros.append(("ENABLE_CUDA_MPI", None))

        # Add MPI paths for CUDA backend
        include_dirs.extend(mpi_include_dirs)
        library_dirs.extend(mpi_lib_dirs)

        # If we found cuFFTMp in a specific subfolder (like math_libs/include), add it
        if cufftmp_inc_found:
            include_dirs.append(cufftmp_inc_found)

        # cuFFTMp libraries
        libraries.extend(["cufftMp", "nvshmem"])
        libraries.extend(mpi_libraries)

    # Custom Build Extension
    class BuildExtensionCuda(build_ext):
        def build_extensions(self):
            # Filter out .cu files to prevent default compiler from failing on them
            for ext in self.extensions:
                cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
                ext.sources = [s for s in ext.sources if not s.endswith(".cu")]

                if cuda_sources:
                    for source in cuda_sources:
                        # Compile .cu file to .o file
                        rel_path = os.path.relpath(source, start=".")
                        obj_rel_path = os.path.splitext(rel_path)[0] + ".o"
                        obj_path = os.path.join(self.build_temp, obj_rel_path)

                        # Ensure directory exists in build_temp
                        os.makedirs(os.path.dirname(obj_path), exist_ok=True)

                        cmd = [ NVCC_PATH, "-c", source, "-o", obj_path,
                            "-std=c++17", "-Xcompiler", "-fPIC", "-arch=native"
                        ]

                        for macro, val in ext.define_macros:
                            cmd.append(f"-D{macro}")

                        # Add Include Paths
                        for inc in ext.include_dirs:
                            cmd.extend(["-I", inc])

                        print(f"Compiling CUDA: {' '.join(cmd)}")
                        self.spawn(cmd)

                        # Link the resulting object file
                        ext.extra_objects.append(obj_path)

            super().build_extensions()

    cmdclass["build_ext"] = BuildExtensionCuda
else:
    print("CUDA not found. Skipping GPU backend.")

# Extension Definition
ext_modules = [
    Extension(
        "anyFFT._core",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        language="c++",
    )
]

setup(
    ext_modules=ext_modules,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    cmdclass=cmdclass,
)