cmake_minimum_required(VERSION 3.21)
project(anyFFT LANGUAGES CXX)

# Python and Pybind11 Setup
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Dynamically find NumPy includes
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Source File Component Lists
set(CORE_SOURCES
    cpp/bindings/module.cpp
)

set(FFTW_SOURCES
    cpp/src/cpu/fftw.cpp
)

set(FFTW_MPI_SOURCES
    cpp/src/cpu/fftw_mpi.cpp
)

set(GPU_CORE_SOURCES
    cpp/src/gpu/gpufft.cu
)

set(GPU_MPI_SOURCES
    cpp/src/gpu/gpufft_dist.cu
)

set(CUDA_MPI_EXTRA_SOURCES
    cpp/src/gpu/cufftmp.cu
)

# Create the Python module
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
set(PYBIND11_LTO_CXX_FLAGS "")
set(PYBIND11_LTO_LINKER_FLAGS "")

pybind11_add_module(_core ${CORE_SOURCES})
target_compile_features(_core PRIVATE cxx_std_17)

set_target_properties(_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

target_include_directories(_core PRIVATE
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

# System Compatibility Fixes
# Disable LTO to prevent GCC/Clang bytecode clashes
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_core PRIVATE "-fno-lto")
    target_link_options(_core PRIVATE "-fno-lto")
endif()

# Fix Conda sysroot for isolated Linux environments
if(DEFINED ENV{CONDA_PREFIX} AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(CONDA_SYSROOT "$ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CONDA_SYSROOT "$ENV{CONDA_PREFIX}/aarch64-conda-linux-gnu/sysroot")
    endif()

    if(EXISTS "${CONDA_SYSROOT}")
        target_link_options(_core PRIVATE "--sysroot=${CONDA_SYSROOT}")
    endif()
endif()

# Build Options
option(ANYFFT_USE_FFTW "Enable FFTW Backend" ON)
option(ANYFFT_USE_CUDA "Enable CUDA Backend" OFF)
option(ANYFFT_USE_HIP  "Enable HIP/ROCm Backend" OFF)
option(ANYFFT_USE_MPI  "Enable MPI Distributed Support" ON)

set(ANYFFT_BACKEND_CONFIGURED FALSE)

# Dependency Detection
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ANYFFT_USE_MPI)
    find_package(MPI)
endif()

# FFTW Backend
if(ANYFFT_USE_FFTW)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(FFTW QUIET IMPORTED_TARGET fftw3 fftw3f)
    endif()

    if(TARGET PkgConfig::FFTW)
        target_sources(_core PRIVATE ${FFTW_SOURCES})
        target_link_libraries(_core PRIVATE PkgConfig::FFTW)
        target_compile_definitions(_core PRIVATE ENABLE_FFTW)
        set(ANYFFT_BACKEND_CONFIGURED TRUE)

        if(MPI_CXX_FOUND AND ANYFFT_USE_MPI)
            pkg_check_modules(FFTW_MPI QUIET IMPORTED_TARGET fftw3_mpi fftw3f_mpi)
            if(TARGET PkgConfig::FFTW_MPI)
                target_sources(_core PRIVATE ${FFTW_MPI_SOURCES})
                target_link_libraries(_core PRIVATE PkgConfig::FFTW_MPI MPI::MPI_CXX)
                target_compile_definitions(_core PRIVATE ENABLE_FFTW_MPI)
            else()
                find_library(FFTW3_MPI_LIB fftw3_mpi)
                find_library(FFTW3F_MPI_LIB fftw3f_mpi)
                if(FFTW3_MPI_LIB)
                    target_sources(_core PRIVATE ${FFTW_MPI_SOURCES})
                    target_link_libraries(_core PRIVATE ${FFTW3_MPI_LIB} ${FFTW3F_MPI_LIB} MPI::MPI_CXX)
                    target_compile_definitions(_core PRIVATE ENABLE_FFTW_MPI)
                endif()
            endif()
        endif()
    else()
        find_library(FFTW3_LIB fftw3)
        find_library(FFTW3F_LIB fftw3f)
        find_path(FFTW_INCLUDE_DIR fftw3.h)

        if(FFTW3_LIB AND FFTW3F_LIB)
            target_sources(_core PRIVATE ${FFTW_SOURCES})
            target_include_directories(_core PRIVATE ${FFTW_INCLUDE_DIR})
            target_link_libraries(_core PRIVATE ${FFTW3_LIB} ${FFTW3F_LIB})
            target_compile_definitions(_core PRIVATE ENABLE_FFTW)
            set(ANYFFT_BACKEND_CONFIGURED TRUE)

            if(MPI_CXX_FOUND AND ANYFFT_USE_MPI)
                find_library(FFTW3_MPI_LIB fftw3_mpi)
                find_library(FFTW3F_MPI_LIB fftw3f_mpi)
                if(FFTW3_MPI_LIB)
                    target_sources(_core PRIVATE ${FFTW_MPI_SOURCES})
                    target_link_libraries(_core PRIVATE ${FFTW3_MPI_LIB} ${FFTW3F_MPI_LIB} MPI::MPI_CXX)
                    target_compile_definitions(_core PRIVATE ENABLE_FFTW_MPI)
                endif()
            endif()
        endif()
    endif()
endif()

# CUDA Backend
if(ANYFFT_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(_core PRIVATE ${GPU_CORE_SOURCES})
    target_link_libraries(_core PRIVATE CUDA::cufft CUDA::cudart)
    target_compile_definitions(_core PRIVATE ENABLE_CUDA)
    set(ANYFFT_BACKEND_CONFIGURED TRUE)

    if(MPI_CXX_FOUND AND ANYFFT_USE_MPI)
        find_library(CUFFTMP_LIB cufftMp)
        find_path(CUFFTMP_INC cufftMp.h)
        if(CUFFTMP_LIB)
            target_sources(_core PRIVATE ${CUDA_MPI_EXTRA_SOURCES} ${GPU_MPI_SOURCES})
            target_include_directories(_core PRIVATE ${CUFFTMP_INC})
            target_link_libraries(_core PRIVATE ${CUFFTMP_LIB} MPI::MPI_CXX)
            target_compile_definitions(_core PRIVATE ENABLE_CUDA_MPI)
        endif()
    endif()
endif()

# HIP / ROCm Backend
if(ANYFFT_USE_HIP)
    enable_language(HIP)

    set_source_files_properties(${GPU_CORE_SOURCES} PROPERTIES LANGUAGE HIP)
    target_sources(_core PRIVATE ${GPU_CORE_SOURCES})

    find_package(hipfft CONFIG REQUIRED)
    find_package(rocfft CONFIG REQUIRED)

    target_link_libraries(_core PRIVATE hip::hipfft roc::rocfft)
    target_compile_definitions(_core PRIVATE __HIP_PLATFORM_AMD__ ENABLE_HIP)
    set(ANYFFT_BACKEND_CONFIGURED TRUE)

    if(MPI_CXX_FOUND AND ANYFFT_USE_MPI)
        set_source_files_properties(${GPU_MPI_SOURCES} PROPERTIES LANGUAGE HIP)
        target_sources(_core PRIVATE ${GPU_MPI_SOURCES})
        target_link_libraries(_core PRIVATE MPI::MPI_CXX)
        target_compile_definitions(_core PRIVATE ENABLE_HIP_MPI)
    endif()
endif()

if(NOT ANYFFT_BACKEND_CONFIGURED)
    message(FATAL_ERROR "No valid FFT backends found. Please check your CMAKE_PREFIX_PATH.")
endif()

install(TARGETS _core DESTINATION anyFFT)
