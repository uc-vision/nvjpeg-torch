#include <vector>
#include <cassert>
#include <memory>
#include <iostream>
#include <cstring>

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <NvJpegDecoder.h>
#include <NvJpegEncoder.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDAHelper.h>



class JpegException : public std::exception {
  int code;
  std::string context;
  

  public:
    JpegException(std::string const& _context, int _code) :
      code(_code), context(_context)
    { }
        
    const char * what () const throw () {
      std::stringstream ss;
      ss << context << ", nvjpeg error " << code;
      return ss.str().c_str();

    }
};


class JpegCoder {
  public:
  std::shared_ptr<NvJPEGEncoder> nv_encoder;

  JpegCoder() {
    nv_encoder.reset(NvJPEGEncoder::createJPEGEncoder("encoder"));
  }


  torch::Tensor encode(torch::Tensor const& y, torch::Tensor const &uv, int quality = 90) {
    py::gil_scoped_release release;

    if (!y.is_contiguous() && !uv.is_contiguous()) {
      throw std::runtime_error("Input tensor must be contiguous");
    }

    if (y.dim() != 2 || uv.dim() != 3) {
      throw std::runtime_error("Input tensor must be 2-dimensional");
    }

    if (y.size(0) != uv.size(1) * 2 || y.size(1) != uv.size(2) * 2) {
      throw std::runtime_error("Input tensor width must be divisible by 2");
    }

    int height = y.size(0);
    int width = y.size(1);

    NvBuffer buffer(V4L2_PIX_FMT_YUV420M, width, height, 0);
    // *const_cast<v4l2_memory*>(&buffer.memory_type) = V4L2_CUDA_MEM_TYPE_DEVICE;


    buffer.planes[0].data = static_cast<unsigned char*>(y.data_ptr());
    buffer.planes[1].data = static_cast<unsigned char*>(uv[0].data_ptr());
    buffer.planes[2].data = static_cast<unsigned char*>(uv[1].data_ptr());

    unsigned long out_buf_size = height * width * 3 / 2;
    auto output = torch::empty({ static_cast<long int>(out_buf_size) }, torch::TensorOptions().dtype(torch::kUInt8));

    auto data = (unsigned char*)output.data_ptr();

    int code = nv_encoder->encodeFromBuffer(buffer, JCS_YCbCr, &data, out_buf_size, quality);
    if (0 != code){
        throw JpegException("Failed to encode jpeg", code);
    }
    
    return output.narrow(0, 0, out_buf_size);

  }



};

  


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto jpeg = py::class_<JpegCoder>(m, "Jpeg");

  py::register_exception<JpegException>(m, "JpegException");

  jpeg.def(py::init<>())
      .def("encode", &JpegCoder::encode)
      .def("__repr__", [](const JpegCoder &a) { return "Jpeg"; });


}
