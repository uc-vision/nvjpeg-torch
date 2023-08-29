import torch
import nvjpeg_cuda

from nvjpeg_cuda import JpegException
from beartype import beartype

class Jpeg():

  Exception = JpegException

  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  # @beartype
  # def encode_y_uv(self, y:torch.Tensor, uv:torch.Tensor,  quality:int=90):
  #   return self.jpeg.encode(y, uv, quality)

  @beartype 
  def encode_yuv_420(self, yuv:torch.Tensor, quality:int=90):
    assert yuv.dim() == 2 and yuv.shape[0] % 3 == 0 and yuv.shape[1] % 2 == 0, \
      f"yuv must be 2 dimensional h*3/2 x w, got {yuv.shape}"
    
    # width = yuv.shape[1]
    # height = yuv.shape[0] // 3 * 2

    # y = yuv[:height]
    # uv = yuv[height:].view(2, height//2, width//2)
    # return self.encode_y_uv(y, uv, quality)


    return self.jpeg.encode(yuv, quality)

