PYTHON   := python3
CXX      := clang++
NVCC     := nvcc

# Configuration
ENABLE_FFTW     ?= 1
ENABLE_MPI      ?= 0
ENABLE_CUDA     ?= 0
ENABLE_CUDA_MPI ?= 0

# Python Config
PY_INC := $(shell $(PYTHON) -m pybind11 --includes) \
          -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
PY_LDFLAGS := $(shell $(PYTHON)-config --ldflags)
EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)

# Base Includes
INC_FLAGS := -Icpp/include -Icpp/src

# Base Flags
CXXFLAGS := -O3 -Wall -fPIC -std=c++17 $(PY_INC) $(INC_FLAGS)
LDFLAGS  :=

# FFTW Configuration
FFTW_ROOT ?= /opt/homebrew/Cellar/fftw/3.3.10_2
FFTW_INC  := $(FFTW_ROOT)/include
FFTW_LIB  := $(FFTW_ROOT)/lib

ifeq ($(ENABLE_FFTW), 1)
    CXXFLAGS += -DENABLE_FFTW -I$(FFTW_INC)
    LDFLAGS  += -L$(FFTW_LIB) -lfftw3 -lfftw3f -lm
endif

# MPI Configuration
ifeq ($(ENABLE_MPI), 1)
    CXX := mpic++
    CXXFLAGS += -DENABLE_FFTW_MPI
    LDFLAGS  += -lfftw3_mpi -lfftw3f_mpi -lmpi
endif

# CUDA Configuration
CUDA_HOME ?= /usr/local/cuda
CUDA_LIB  := $(CUDA_HOME)/lib64

ifeq ($(ENABLE_CUDA), 1)
    CXXFLAGS += -DENABLE_CUDA

    # NVCC Flags
    NVCCFLAGS := -O3 -std=c++17 -Xcompiler -fPIC -DENABLE_CUDA \
                 $(PY_INC) $(INC_FLAGS) -arch=native

    # Add Serial CUDA Source
    SOURCES_CU := cpp/src/cufft/cufft_serial.cu

    # Add Parallel CUDA Source
    ifeq ($(ENABLE_CUDA_MPI), 1)
        CXXFLAGS += -DENABLE_CUDA_MPI
        NVCCFLAGS += -DENABLE_CUDA_MPI
        SOURCES_CU += cpp/src/cufft/cufft_mpi.cu

        # Link against cuFFTMp and NVSHMEM if needed
        LDFLAGS += -lcufftMp -lnvshmem
    endif

    OBJS_CU := $(SOURCES_CU:.cu=.o)

    # Link CUDA libraries
    LDFLAGS += -L$(CUDA_LIB) -lcudart -lcufft
else
    SOURCES_CU :=
    OBJS_CU    :=
endif

# macOS Linker Fix
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
    LDFLAGS += -undefined dynamic_lookup
endif

# Source Definitions
SOURCES_CPP := cpp/src/module.cpp

ifeq ($(ENABLE_FFTW), 1)
    SOURCES_CPP += cpp/src/fftw/fftw_serial.cpp
endif

ifeq ($(ENABLE_MPI), 1)
    SOURCES_CPP += cpp/src/fftw/fftw_mpi.cpp
endif

OBJS_CPP := $(SOURCES_CPP:.cpp=.o)

# Target Output
TARGET_DIR := src/anyFFT
TARGET     := $(TARGET_DIR)/_core$(EXT_SUFFIX)

# Build Rules
all: $(TARGET)

# Link
$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	@mkdir -p $(TARGET_DIR)
	$(CXX) -shared $(OBJS_CPP) $(OBJS_CU) -o $(TARGET) $(LDFLAGS) $(PY_LDFLAGS)

# Compile C++
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(TARGET)
	rm -f cpp/src/fftw/*.o cpp/src/cufft/*.o cpp/src/*.o
	rm -rf build/ dist/ src/*.egg-info
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

.PHONY: all clean