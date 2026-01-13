import os
import sys
import shutil
import subprocess
import sysconfig
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11
import numpy

# Configuration
NAME = "anyFFT"
VERSION = "0.0.2"
DESCRIPTION = "Serial and Parallel FFT bindings with pybind11 for CPU/GPU"

# Dynamic Path Discovery
# Try to find FFTW in standard locations or via env variable
FFTW_ROOT = os.environ.get("FFTW_ROOT", None)

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
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC_PATH = shutil.which("nvcc")
HAS_CUDA = NVCC_PATH is not None

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

define_macros = []

# Base includes
include_dirs = [
    pybind11.get_include(),
    numpy.get_include(),
    sysconfig.get_path("include"),
    sysconfig.get_path("platinclude"),
    "cpp"
    "cpp/fft",
    "cpp/fft/backends",
]

sources = ["cpp/bindings/module.cpp"]
libraries = ["m"]
library_dirs = []

# Conditional Backend Inclusion
if HAS_FFTW:
    print(f"FFTW found at {FFTW_INC}. Enabling CPU support.")
    sources.append("cpp/fft/backends/fftw_serial.cpp")
    define_macros.append(("ENABLE_FFTW", None))
    include_dirs.append(FFTW_INC)
    library_dirs.append(FFTW_LIB)
    libraries.extend(["fftw3", "fftw3f"])
else:
    print("FFTW not found. Skipping CPU backend.")

cmdclass = {}
if HAS_CUDA:
    print(f"CUDA found at {NVCC_PATH}. Enabling GPU support.")
    sources.append("cpp/fft/backends/cufft_serial.cu")
    define_macros.append(("ENABLE_CUDA", None))
    include_dirs.append(os.path.join(CUDA_HOME, "include"))

    cuda_lib64 = os.path.join(CUDA_HOME, "lib64")
    if os.path.isdir(cuda_lib64):
        library_dirs.append(cuda_lib64)
    else:
        library_dirs.append(os.path.join(CUDA_HOME, "lib"))

    libraries.extend(["cufft", "cudart"])

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
                        obj_path = os.path.splitext(source)[0] + ".o"

                        cmd = [ NVCC_PATH, "-c", source, "-o", obj_path,
                            "-std=c++17", "-Xcompiler", "-fPIC", "-DENABLE_CUDA"
                        ]

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
        f"{NAME}.anyFFT",
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
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    ext_modules=ext_modules,
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    cmdclass=cmdclass,
    zip_safe=False,
)