from setuptools import setup, Extension
import pybind11
import numpy
import sys
import os

# paths to FFTW
FFTW_INC = "/opt/homebrew/Cellar/fftw/3.3.10_2/include"
FFTW_LIB = "/opt/homebrew/Cellar/fftw/3.3.10_2/lib"

# source files
sources = [
    "src/bindings/module.cpp",
    "src/fft/fft.cpp",
    "src/fft/backends/fftw_serial.cpp",
]

# extension module
ext_modules = [
    Extension(
        "anyFFT.anyFFT",  # module name inside package anyFFT
        sources=sources,
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            FFTW_INC,
            "src/fft",
            "src/fft/backends",
        ],
        library_dirs=[FFTW_LIB],
        libraries=["fftw3", "fftw3f", "m"],
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-undefined", "dynamic_lookup"],
        language="c++",
    )
]

setup(
    name="anyFFT",
    version="0.0.1",
    author="5hashN",
    description="Serial and Parallel FFT bindings with pybind11 for CPU/GPU",
    ext_modules=ext_modules,
    packages=["anyFFT"],
    package_dir={"anyFFT": "anyFFT"},
    zip_safe=False,
)
