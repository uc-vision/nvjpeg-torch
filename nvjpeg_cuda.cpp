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

#include <nvbuf_utils.h>
#include <nvbufsurface.h>

#include "NvBufSurface.h"


#include "NvUtils.h"

#include <cudaEGL.h>
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

typedef struct
{
    unsigned int v4l2_pixfmt;
    NvBufSurfaceColorFormat nvbuff_color;
} nv_color_fmt;



static nv_color_fmt nvcolor_fmt[] =
{
    /* TODO: add more pixel format mapping */
    {V4L2_PIX_FMT_UYVY, NVBUF_COLOR_FORMAT_UYVY},
    {V4L2_PIX_FMT_VYUY, NVBUF_COLOR_FORMAT_VYUY},
    {V4L2_PIX_FMT_YUYV, NVBUF_COLOR_FORMAT_YUYV},
    {V4L2_PIX_FMT_YVYU, NVBUF_COLOR_FORMAT_YVYU},
    {V4L2_PIX_FMT_GREY, NVBUF_COLOR_FORMAT_GRAY8},
    {V4L2_PIX_FMT_YUV420M, NVBUF_COLOR_FORMAT_YUV420},
};

static NvBufSurfaceColorFormat
get_nvbuff_color_fmt(unsigned int v4l2_pixfmt)
{
    unsigned i;

    for (i = 0; i < sizeof(nvcolor_fmt) / sizeof(nvcolor_fmt[0]); i++)
    {
        if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
            return nvcolor_fmt[i].nvbuff_color;
    }

    return NVBUF_COLOR_FORMAT_INVALID;
}



class JpegCoder {
  public:

  std::shared_ptr<NvJPEGEncoder> nv_encoder;
  std::mutex mtx; 

  int dmabuf_fd = -1;

  JpegCoder() {
    nv_encoder.reset(NvJPEGEncoder::createJPEGEncoder("encoder"));




  }




  // torch::Tensor encode_egl(torch::Tensor const& yuv, int quality = 90) {
  //   py::gil_scoped_release release;

  //   // if (!y.is_contiguous() && !uv.is_contiguous()) {
  //   //   throw std::runtime_error("Input tensor must be contiguous");
  //   // }

  //   // if (y.dim() != 2 || uv.dim() != 3) {
  //   //   throw std::runtime_error("Input tensor must be 2-dimensional");
  //   // }

  //   // if (y.size(0) != uv.size(1) * 2 || y.size(1) != uv.size(2) * 2) {
  //   //   throw std::runtime_error("Input tensor width must be divisible by 2");
  //   // }

  //   int height = yuv.size(0) / 3 * 2;
  //   int width = yuv.size(1);

  //   int dmabuf_fd = 0;


  //   NvBufSurf::NvCommonAllocateParams params = {0};
  //   params.memType =  NVBUF_MEM_SURFACE_ARRAY;
  //   params.width = width;
  //   params.height = height;
  //   // params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
  //   params.layout = NVBUF_LAYOUT_PITCH;

  //   params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);
  //   params.memtag = NvBufSurfaceTag_NONE;


  //   if (NvBufSurf::NvAllocate(&params, 1, &dmabuf_fd))
  //     throw std::runtime_error("Failed to create NvBuffer");

  //   NvBufSurface *surf;
  //   NvBufSurfaceFromFd(dmabuf_fd, (void**)&surf);

  //   NvBufSurfaceMemSet (surf, 0, 0, 128);
  //   NvBufSurfaceSyncForDevice(surf, -1, -1);


  //   NvBufSurfaceMapEglImage(surf, 0);

  //   CUgraphicsResource resource;
  //   CUeglFrame eglFrame;


  //   cuGraphicsEGLRegisterImage(&resource, surf->surfaceList[0].mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  //   cuGraphicsResourceGetMappedEglFrame(&eglFrame, resource, 0, 0);


  //   cudaMemcpy( eglFrame.frame.pPitch[0], (void*)yuv.data_ptr(), yuv.size(0) * yuv.size(1), cudaMemcpyDeviceToDevice);

  //   cuGraphicsUnregisterResource(resource);

  //   NvBufSurfaceSyncForDevice(surf, -1, -1);
  //   NvBufSurfaceUnMapEglImage(surf, 0);


  //   unsigned long out_buf_size = height * width * 3 / 2;
  //   auto output = torch::empty({ static_cast<long int>(out_buf_size) }, torch::TensorOptions().dtype(torch::kUInt8));


  //   auto data = (unsigned char*)output.data_ptr();

  //   std::cout << "Here!!!" << std::endl;

  //   mtx.lock();
  //   // int code = nv_encoder->encodeFromBuffer(buffer, JCS_YCbCr, &data, out_buf_size, quality);
  //   int code = nv_encoder->encodeFromFd(dmabuf_fd, JCS_YCbCr, &data, out_buf_size, quality);

  //   mtx.unlock();



  //   if (0 != code){
  //       throw JpegException("Failed to encode jpeg", code);
  //   }

    
  //   return output.narrow(0, 0, out_buf_size);
  // }

  void allocate_fd(int width, int height) {
    NvBufSurf::NvCommonAllocateParams params = {0};
    params.memType =  NVBUF_MEM_SURFACE_ARRAY;
    params.width = width;
    params.height = height;
    params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
    // params.layout = NVBUF_LAYOUT_PITCH;

    params.colorFormat = get_nvbuff_color_fmt(V4L2_PIX_FMT_YUV420M);
    params.memtag = NvBufSurfaceTag_NONE;

    if (NvBufSurf::NvAllocate(&params, 1, &dmabuf_fd))
      throw std::runtime_error("Failed to create NvBuffer");
  }


  torch::Tensor encode(torch::Tensor const& yuv, int quality = 90) {
    py::gil_scoped_release release;

    int height = yuv.size(0) / 3 * 2;
    int width = yuv.size(1);

    auto y = yuv.narrow(0, 0, height);
    auto uv = yuv.narrow(0, height, yuv.size(0) - height).view({2, height / 2, width / 2});


    mtx.lock();


    if (dmabuf_fd == -1) {
      allocate_fd(width, height);
    }


    if (Raw2NvBuffer((unsigned char*)y.data_ptr(), 0, width, height, dmabuf_fd)) {
      throw std::runtime_error("Failed to copy Y plane to NvBuffer");
    }

    for (int i = 0; i < 2; ++i) {
      if (Raw2NvBuffer((unsigned char*)uv[i].data_ptr(), i + 1, width / 2, height / 2, dmabuf_fd)) {
        throw std::runtime_error("Failed to copy U plane to NvBuffer");
      }
    }

    
    unsigned long out_buf_size = height * width * 3 / 2;
    auto output = torch::empty({ static_cast<long int>(out_buf_size) }, torch::TensorOptions().dtype(torch::kUInt8));

    auto data = (unsigned char*)output.data_ptr();

    // int code = nv_encoder->encodeFromBuffer(buffer, JCS_YCbCr, &data, out_buf_size, quality);
    int code = nv_encoder->encodeFromFd(dmabuf_fd, JCS_YCbCr, &data, out_buf_size, quality);

    mtx.unlock();


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
