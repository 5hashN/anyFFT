PYTHON   := python3
CXX      := clang++
FFTW_INC := /opt/homebrew/Cellar/fftw/3.3.10_2/include
FFTW_LIB := /opt/homebrew/Cellar/fftw/3.3.10_2/lib

# all source files
SOURCES := src/bindings/module.cpp \
           src/fft/fft.cpp \
           src/fft/backends/fftw_serial.cpp

# target shared library
TARGET := anyFFT/anyFFT$(shell $(PYTHON)-config --extension-suffix)

# compiler flags
CXXFLAGS := -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup \
            $(shell $(PYTHON) -m pybind11 --includes) \
            -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())") \
            -I$(FFTW_INC) \
            -Isrc/fft \
            -Isrc/fft/backends

# linker flags
LDFLAGS := -L$(FFTW_LIB) -lfftw3 -lfftw3f -lm $(shell $(PYTHON)-config --ldflags)

# default target
all: $(TARGET)

# build shared library into python/ folder
$(TARGET): $(SOURCES)
	@mkdir -p python
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

# clean build artifacts
clean:
	rm -f anyFFT/anyFFT*.so
