#include "fftw_serial.hpp"

// C2C (Complex-to-Complex)
class FFTW_C2C : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t N_;
    std::string dtype_;
    void* plan_forward_;
    void* plan_backward_;

public:
    FFTW_C2C(int ndim, const std::vector<int>& shape,
             py::array in_arr, py::array out_arr, std::string dtype)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype),
          plan_forward_(nullptr), plan_backward_(nullptr)
    {
        for (int i : shape_) N_ *= i;

        void* in_ptr = in_arr.mutable_data();
        void* out_ptr = out_arr.mutable_data();

        // Complex128
        if (dtype_ == "complex128") {
            fftw_complex* i_p = reinterpret_cast<fftw_complex*>(in_ptr);
            fftw_complex* o_p = reinterpret_cast<fftw_complex*>(out_ptr);

            if (ndim_ == 1) {
                plan_forward_ = fftw_plan_dft_1d(shape_[0], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftw_plan_dft_1d(shape_[0], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
                plan_forward_ = fftw_plan_dft_2d(shape_[0], shape_[1], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftw_plan_dft_2d(shape_[0], shape_[1], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            } else if (ndim_ == 3) {
                plan_forward_ = fftw_plan_dft_3d(shape_[0], shape_[1], shape_[2], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftw_plan_dft_3d(shape_[0], shape_[1], shape_[2], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            }
        }
        // Complex64
        else if (dtype_ == "complex64") {
            fftwf_complex* i_p = reinterpret_cast<fftwf_complex*>(in_ptr);
            fftwf_complex* o_p = reinterpret_cast<fftwf_complex*>(out_ptr);

            if (ndim_ == 1) {
                plan_forward_ = fftwf_plan_dft_1d(shape_[0], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftwf_plan_dft_1d(shape_[0], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
                plan_forward_ = fftwf_plan_dft_2d(shape_[0], shape_[1], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftwf_plan_dft_2d(shape_[0], shape_[1], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            } else if (ndim_ == 3) {
                plan_forward_ = fftwf_plan_dft_3d(shape_[0], shape_[1], shape_[2], i_p, o_p, FFTW_FORWARD, FFTW_ESTIMATE);
                plan_backward_ = fftwf_plan_dft_3d(shape_[0], shape_[1], shape_[2], i_p, o_p, FFTW_BACKWARD, FFTW_ESTIMATE);
            }
        }
    }

    void forward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();

        if (dtype_ == "complex128") {
            fftw_execute_dft((fftw_plan)plan_forward_,
                             (fftw_complex*)in.mutable_data(),
                             (fftw_complex*)out.mutable_data());
        } else {
            fftwf_execute_dft((fftwf_plan)plan_forward_,
                              (fftwf_complex*)in.mutable_data(),
                              (fftwf_complex*)out.mutable_data());
        }
    }

    void backward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();

        if (dtype_ == "complex128") {
            fftw_execute_dft((fftw_plan)plan_backward_,
                             (fftw_complex*)in.mutable_data(),
                             (fftw_complex*)out.mutable_data());

            // Normalize (Complex has 2 doubles per element, so we iterate over 2*size)
            double* buf = (double*)out.mutable_data();
            ssize_t total = out.size() * 2;
            for(ssize_t i=0; i < total; ++i) buf[i] /= N_;
        }
        else {
            fftwf_execute_dft((fftwf_plan)plan_backward_,
                              (fftwf_complex*)in.mutable_data(),
                              (fftwf_complex*)out.mutable_data());

            float* buf = (float*)out.mutable_data();
            ssize_t total = out.size() * 2;
            for(ssize_t i=0; i < total; ++i) buf[i] /= N_;
        }
    }

    ~FFTW_C2C() {
        if (dtype_ == "complex128") {
            if (plan_forward_) fftw_destroy_plan((fftw_plan)plan_forward_);
            if (plan_backward_) fftw_destroy_plan((fftw_plan)plan_backward_);
        } else {
            if (plan_forward_) fftwf_destroy_plan((fftwf_plan)plan_forward_);
            if (plan_backward_) fftwf_destroy_plan((fftwf_plan)plan_backward_);
        }
    }
};

// R2C Out-of-Place
class FFTW_R2C_OutPlace : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_R2C_OutPlace(int ndim, const std::vector<int>& shape,
                  py::array real_in, py::array complex_out, std::string dtype)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
        for (int i : shape_) N_ *= i;

        if (dtype_ == "float64") {
            double* r_ptr = static_cast<double*>(real_in.mutable_data());
            fftw_complex* c_ptr = reinterpret_cast<fftw_complex*>(complex_out.mutable_data());

            if (ndim_ == 1) {
                plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], c_ptr, r_ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
                plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0], shape_[1], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0], shape_[1], c_ptr, r_ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 3) {
                plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0], shape_[1], shape_[2], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0], shape_[1], shape_[2], c_ptr, r_ptr, FFTW_ESTIMATE);
            }
        }
        else if (dtype_ == "float32") {
            float* r_ptr = static_cast<float*>(real_in.mutable_data());
            fftwf_complex* c_ptr = reinterpret_cast<fftwf_complex*>(complex_out.mutable_data());

            if (ndim_ == 1) {
                plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], c_ptr, r_ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 2) {
                plan_r2c_ = fftwf_plan_dft_r2c_2d(shape_[0], shape_[1], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_2d(shape_[0], shape_[1], c_ptr, r_ptr, FFTW_ESTIMATE);
            } else if (ndim_ == 3) {
                plan_r2c_ = fftwf_plan_dft_r2c_3d(shape_[0], shape_[1], shape_[2], r_ptr, c_ptr, FFTW_ESTIMATE);
                plan_c2r_ = fftwf_plan_dft_c2r_3d(shape_[0], shape_[1], shape_[2], c_ptr, r_ptr, FFTW_ESTIMATE);
            }
        }
    }

    void forward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();
        if (dtype_ == "float64") fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)in.mutable_data(), (fftw_complex*)out.mutable_data());
        else fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)in.mutable_data(), (fftwf_complex*)out.mutable_data());
    }

    void backward(py::object in_obj, py::object out_obj) override {
        py::array in = in_obj.cast<py::array>();
        py::array out = out_obj.cast<py::array>();
        if (dtype_ == "float64") {
            fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)in.mutable_data(), (double*)out.mutable_data());
            double* buf = (double*)out.mutable_data();
            for(ssize_t i=0; i<out.size(); ++i) buf[i] /= N_;
        } else {
            fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)in.mutable_data(), (float*)out.mutable_data());
            float* buf = (float*)out.mutable_data();
            for(ssize_t i=0; i<out.size(); ++i) buf[i] /= N_;
        }
    }

    ~FFTW_R2C_OutPlace() {
        if (dtype_ == "float64") { if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_); if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_); }
        else { if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_); if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_); }
    }
};

// R2C In-Place
class FFTW_R2C_InPlace : public FFTBase {
    std::vector<int> shape_;
    int ndim_;
    ssize_t N_;
    std::string dtype_;
    void* plan_r2c_;
    void* plan_c2r_;

public:
    FFTW_R2C_InPlace(int ndim, const std::vector<int>& shape, py::array data, std::string dtype)
        : shape_(shape), ndim_(ndim), N_(1), dtype_(dtype), plan_r2c_(nullptr), plan_c2r_(nullptr)
    {
         for (int i : shape_) N_ *= i;
         void* ptr = data.mutable_data();

         if (dtype_ == "float64") {
             double* r_ptr = static_cast<double*>(ptr);
             fftw_complex* c_ptr = reinterpret_cast<fftw_complex*>(ptr);

             if (ndim_ == 1) {
                 plan_r2c_ = fftw_plan_dft_r2c_1d(shape_[0], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftw_plan_dft_c2r_1d(shape_[0], c_ptr, r_ptr, FFTW_ESTIMATE);
             } else if (ndim_ == 2) {
                 plan_r2c_ = fftw_plan_dft_r2c_2d(shape_[0], shape_[1], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftw_plan_dft_c2r_2d(shape_[0], shape_[1], c_ptr, r_ptr, FFTW_ESTIMATE);
             } else if (ndim_ == 3) {
                 plan_r2c_ = fftw_plan_dft_r2c_3d(shape_[0], shape_[1], shape_[2], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftw_plan_dft_c2r_3d(shape_[0], shape_[1], shape_[2], c_ptr, r_ptr, FFTW_ESTIMATE);
             }
         }
         else if (dtype_ == "float32") {
             float* r_ptr = static_cast<float*>(ptr);
             fftwf_complex* c_ptr = reinterpret_cast<fftwf_complex*>(ptr);

             if (ndim_ == 1) {
                 plan_r2c_ = fftwf_plan_dft_r2c_1d(shape_[0], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftwf_plan_dft_c2r_1d(shape_[0], c_ptr, r_ptr, FFTW_ESTIMATE);
             } else if (ndim_ == 2) {
                 plan_r2c_ = fftwf_plan_dft_r2c_2d(shape_[0], shape_[1], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftwf_plan_dft_c2r_2d(shape_[0], shape_[1], c_ptr, r_ptr, FFTW_ESTIMATE);
             } else if (ndim_ == 3) {
                 plan_r2c_ = fftwf_plan_dft_r2c_3d(shape_[0], shape_[1], shape_[2], r_ptr, c_ptr, FFTW_ESTIMATE);
                 plan_c2r_ = fftwf_plan_dft_c2r_3d(shape_[0], shape_[1], shape_[2], c_ptr, r_ptr, FFTW_ESTIMATE);
             }
         }
    }

    void forward(py::object in, py::object out) override {
        py::array arr = in.cast<py::array>();
        void* ptr = arr.mutable_data();
        if (dtype_ == "float64") fftw_execute_dft_r2c((fftw_plan)plan_r2c_, (double*)ptr, (fftw_complex*)ptr);
        else fftwf_execute_dft_r2c((fftwf_plan)plan_r2c_, (float*)ptr, (fftwf_complex*)ptr);
    }

    void backward(py::object in, py::object out) override {
        py::array arr = in.cast<py::array>();
        void* ptr = arr.mutable_data();
        ssize_t total_elements = arr.size();
        if (arr.dtype().kind() == 'c') total_elements *= 2;

        if (dtype_ == "float64") {
            fftw_execute_dft_c2r((fftw_plan)plan_c2r_, (fftw_complex*)ptr, (double*)ptr);
            double* buf = (double*)ptr;
            for(ssize_t i=0; i < total_elements; ++i) buf[i] /= N_;
        } else {
            fftwf_execute_dft_c2r((fftwf_plan)plan_c2r_, (fftwf_complex*)ptr, (float*)ptr);
            float* buf = (float*)ptr;
            for(ssize_t i=0; i < total_elements; ++i) buf[i] /= N_;
        }
    }

    ~FFTW_R2C_InPlace() {
        if (dtype_ == "float64") { if(plan_r2c_) fftw_destroy_plan((fftw_plan)plan_r2c_); if(plan_c2r_) fftw_destroy_plan((fftw_plan)plan_c2r_); }
        else { if(plan_r2c_) fftwf_destroy_plan((fftwf_plan)plan_r2c_); if(plan_c2r_) fftwf_destroy_plan((fftwf_plan)plan_c2r_); }
    }
};

// The Wrapper Constructor
FFTW_SERIAL::FFTW_SERIAL(int ndim, const std::vector<int>& shape,
                         py::array real_in, py::array complex_out,
                         const std::string& dtype)
{
    // Check for Complex-to-Complex (C2C)
    if (dtype == "complex128" || dtype == "complex64") {
        impl_ = std::make_unique<FFTW_C2C>(ndim, shape, real_in, complex_out, dtype);
    }
    // Check for Real-to-Complex In-Place
    else if (real_in.ptr() == complex_out.ptr()) {
        impl_ = std::make_unique<FFTW_R2C_InPlace>(ndim, shape, real_in, dtype);
    }
    // Default to Real-to-Complex Out-of-Place
    else {
        impl_ = std::make_unique<FFTW_R2C_OutPlace>(ndim, shape, real_in, complex_out, dtype);
    }
}

void FFTW_SERIAL::forward(py::object in, py::object out) { impl_->forward(in, out); }
void FFTW_SERIAL::backward(py::object in, py::object out) { impl_->backward(in, out); }
FFTW_SERIAL::~FFTW_SERIAL() = default;