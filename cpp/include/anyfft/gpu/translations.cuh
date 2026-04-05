/*
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
*/

#pragma once

#if defined(ENABLE_HIP)

    #include <hip/hip_runtime.h>
    #include <hipfft/hipfft.h>

    // Core Types
    typedef hipfftHandle        gpufftHandle;
    typedef hipfftComplex       gpufftComplex;
    typedef hipfftDoubleComplex gpufftDoubleComplex;
    typedef hipfftReal          gpufftReal;
    typedef hipfftDoubleReal    gpufftDoubleReal;
    typedef hipfftResult        gpufftResult;
    typedef hipfftType          gpufftType;
    typedef hipError_t          gpufftError_t;

    // Constants
    #define GPUFFT_SUCCESS         HIPFFT_SUCCESS
    #define GPUFFT_RUNTIME_SUCCESS hipSuccess
    #define GPUFFT_C2C             HIPFFT_C2C
    #define GPUFFT_Z2Z             HIPFFT_Z2Z
    #define GPUFFT_R2C             HIPFFT_R2C
    #define GPUFFT_C2R             HIPFFT_C2R
    #define GPUFFT_D2Z             HIPFFT_D2Z
    #define GPUFFT_Z2D             HIPFFT_Z2D
    #define GPUFFT_FORWARD         HIPFFT_FORWARD
    #define GPUFFT_INVERSE         HIPFFT_BACKWARD

    // Memory & Runtime Aliases
    #define gpufftMalloc             hipMalloc
    #define gpufftFree               hipFree
    #define gpufftMemcpy             hipMemcpy
    #define gpufftMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpufftMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpufftDeviceSynchronize  hipDeviceSynchronize
    #define gpufftGetErrorString     hipGetErrorString

    // FFT Function Aliases
    #define gpufftCreate   hipfftCreate
    #define gpufftPlan1d   hipfftPlan1d
    #define gpufftPlan2d   hipfftPlan2d
    #define gpufftPlan3d   hipfftPlan3d
    #define gpufftPlanMany hipfftPlanMany
    #define gpufftDestroy  hipfftDestroy
    #define gpufftExecC2C  hipfftExecC2C
    #define gpufftExecZ2Z  hipfftExecZ2Z
    #define gpufftExecR2C  hipfftExecR2C
    #define gpufftExecC2R  hipfftExecC2R
    #define gpufftExecD2Z  hipfftExecD2Z
    #define gpufftExecZ2D  hipfftExecZ2D

    #if defined(ENABLE_HIP_DIST)
        #include <rocshmem/rocshmem.hpp>

        // ROCSHMEM Aliases
        #define gpushmem_init          rocshmem_init
        #define gpushmem_init_thread   rocshmem_init_thread
        #define gpushmem_finalize      rocshmem_finalize
        #define gpushmem_malloc        rocshmem_malloc
        #define gpushmem_free          rocshmem_free
        #define gpushmem_barrier_all   rocshmem_barrier_all
    #endif

#elif defined(ENABLE_CUDA)

    #include <cuda_runtime.h>
    #include <cufft.h>

    // Core Types
    typedef cufftHandle        gpufftHandle;
    typedef cufftComplex       gpufftComplex;
    typedef cufftDoubleComplex gpufftDoubleComplex;
    typedef cufftReal          gpufftReal;
    typedef cufftDoubleReal    gpufftDoubleReal;
    typedef cufftResult        gpufftResult;
    typedef cufftType          gpufftType;
    typedef cudaError_t        gpufftError_t;

    // Constants
    #define GPUFFT_SUCCESS         CUFFT_SUCCESS
    #define GPUFFT_RUNTIME_SUCCESS cudaSuccess
    #define GPUFFT_C2C             CUFFT_C2C
    #define GPUFFT_Z2Z             CUFFT_Z2Z
    #define GPUFFT_R2C             CUFFT_R2C
    #define GPUFFT_C2R             CUFFT_C2R
    #define GPUFFT_D2Z             CUFFT_D2Z
    #define GPUFFT_Z2D             CUFFT_Z2D
    #define GPUFFT_FORWARD         CUFFT_FORWARD
    #define GPUFFT_INVERSE         CUFFT_INVERSE

    // Memory & Runtime Aliases
    #define gpufftMalloc             cudaMalloc
    #define gpufftFree               cudaFree
    #define gpufftMemcpy             cudaMemcpy
    #define gpufftMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpufftMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpufftDeviceSynchronize  cudaDeviceSynchronize
    #define gpufftGetErrorString     cudaGetErrorString

    // FFT Function Aliases
    #define gpufftCreate   cufftCreate
    #define gpufftPlan1d   cufftPlan1d
    #define gpufftPlan2d   cufftPlan2d
    #define gpufftPlan3d   cufftPlan3d
    #define gpufftPlanMany cufftPlanMany
    #define gpufftDestroy  cufftDestroy
    #define gpufftExecC2C  cufftExecC2C
    #define gpufftExecZ2Z  cufftExecZ2Z
    #define gpufftExecR2C  cufftExecR2C
    #define gpufftExecC2R  cufftExecC2R
    #define gpufftExecD2Z  cufftExecD2Z
    #define gpufftExecZ2D  cufftExecZ2D

    #if defined(ENABLE_CUDA_DIST)
        #include <cufftMp.h>
        #include <nvshmem.h>
        #include <nvshmemx.h>

        // NVSHMEM Aliases
        #define gpushmem_init          nvshmem_init
        #define gpushmem_init_thread   nvshmem_init_thread
        #define gpushmem_finalize      nvshmem_finalize
        #define gpushmem_malloc        nvshmem_malloc
        #define gpushmem_free          nvshmem_free
        #define gpushmem_barrier_all   nvshmem_barrier_all

        // cuFFTMp Types
        typedef cudaLibXtDesc          gpufftLibXtDesc;
        typedef cufftXtSubFormat       gpufftXtSubFormat;

        // cuFFTMp Function Aliases
        #define gpufftMpAttachComm     cufftMpAttachComm
        #define gpufftXtMakePlanMany   cufftXtMakePlanMany
        #define gpufftXtSetGPUs        cufftXtSetGPUs
        #define gpufftXtMalloc         cufftXtMalloc
        #define gpufftXtMemcpy         cufftXtMemcpy
        #define gpufftXtFree           cufftXtFree
        #define gpufftXtExec           cufftXtExec
        #define gpufftXtExecDescriptor cufftXtExecDescriptor
    #endif

#else
    #error "Backend not specified. Please define either ENABLE_CUDA or ENABLE_HIP."
#endif

#include <iostream>
#include <cstdlib>

inline void gpufftAssert(gpufftResult code, const char *file, int line, bool abort = true) {
    if (code != GPUFFT_SUCCESS) {
        std::cerr << "gpuFFT Error: Code " << code << " at " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

#define GPUFFT_CHECK(ans) { gpufftAssert((ans), __FILE__, __LINE__); }
