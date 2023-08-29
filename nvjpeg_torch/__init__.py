import torch
import nvjpeg_cuda

from nvjpeg_cuda import JpegException
from beartype import beartype

class Jpeg():

  Exception = JpegException

  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  @beartype
  def encode_yuv_420(self, yuv:torch.Tensor,  quality:int=90):

    height = yuv.shape[0] * 2 // 3
    width = yuv.shape[1]

    y = yuv[:height]
    uv = yuv[height:].view(2, height//2, width//2)

    return self.jpeg.encode(y, uv, quality)


