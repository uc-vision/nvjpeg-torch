import torch
import nvjpeg_cuda

from nvjpeg_cuda import JpegException
from beartype import beartype

class Jpeg():

  Exception = JpegException

  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  @beartype
  def encode_yuv(self, y:torch.Tensor, uv:torch.Tensor,  quality:int=90):
    return self.jpeg.encode(y, uv, quality)


