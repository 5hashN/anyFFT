PYTHON   := python3
CXX      := clang++
NVCC     := nvcc

# Configuration
ENABLE_MPI ?= 0
ENABLE_CUDA_MPI ?= 0
USE_CUDA ?= 0

ifeq ($(ENABLE_MPI), 1)
    # Switch compiler to MPI wrapper
    CXX := mpic++

    # Add MPI Macro
    CXXFLAGS += -DENABLE_FFTW_MPI

    # Link FFTW MPI libraries
    LDFLAGS += -lfftw3_mpi -lfftw3f_mpi -lmpi
endif

# FFTW Paths
FFTW_INC := /opt/homebrew/Cellar/fftw/3.3.10_2/include
FFTW_LIB := /opt/homebrew/Cellar/fftw/3.3.10_2/lib

# Python Config
PY_INC := $(shell $(PYTHON) -m pybind11 --includes) \
          -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
PY_LDFLAGS := $(shell $(PYTHON)-config --ldflags)
EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

# Base C++ Flags (Compile-time flags only)
CXXFLAGS := -O3 -Wall -fPIC -std=c++17 \
            $(PY_INC) -I$(FFTW_INC) -Icpp/fft -Icpp/fft/backends

# Base Linker Flags
LDFLAGS := -L$(FFTW_LIB) -lfftw3 -lfftw3f -lm

# macOS specific linker adjustment
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
    # This is required on macOS to link against Python symbols found at runtime
    LDFLAGS += -undefined dynamic_lookup
endif

# Detect or set CUDA_HOME (Defaults to /usr/local/cuda)
CUDA_HOME ?= /usr/local/cuda
CUDA_LIB  := $(CUDA_HOME)/lib64

ifeq ($(USE_CUDA), 1)
    # Define Macro so C++ code knows to include CUDA headers
    CXXFLAGS += -DENABLE_CUDA

    # NVCC Flags
    NVCCFLAGS := -O3 -std=c++17 -Xcompiler -fPIC -DENABLE_CUDA \
                 $(PY_INC) -Icpp/fft -Icpp/fft/backends

    # Add CUDA sources and libraries
    SOURCES_CU := cpp/fft/backends/cufft_serial.cu

    # cuFFTMp
    ifeq ($(ENABLE_CUDA_MPI), 1)
        SOURCES_CU += cpp/fft/backends/cufft_mpi.cu
    endif

    OBJS_CU    := $(SOURCES_CU:.cu=.o)

    # Add CUDA library paths to linker
    LDFLAGS    += -L$(CUDA_LIB) -lcudart -lcufft
else
    SOURCES_CU :=
    OBJS_CU    :=
endif

# Source Definitions
SOURCES_CPP := cpp/bindings/module.cpp \
               cpp/fft/backends/fftw_serial.cpp

# Add MPI Source
ifeq ($(ENABLE_MPI), 1)
    SOURCES_CPP += cpp/fft/backends/fftw_mpi.cpp
endif

OBJS_CPP    := $(SOURCES_CPP:.cpp=.o)

# Target Output
TARGET := anyFFT/anyFFT$(EXT_SUFFIX)

# Build Rules
all: $(TARGET)

# Link everything together
# We append PY_LDFLAGS here as well
$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	@mkdir -p anyFFT
	$(CXX) -shared $(OBJS_CPP) $(OBJS_CU) -o $(TARGET) $(LDFLAGS) $(PY_LDFLAGS)

# Compile C++ files to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files to .o (Only if USE_CUDA=1)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(TARGET)
	rm -f cpp/fft/backends/*.o cpp/bindings/*.o cpp/fft/*.o

	rm -rf build/
	rm -rf dist/
	rm -rf python/*.egg-info

	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +