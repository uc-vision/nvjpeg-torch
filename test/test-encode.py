
from os import path
import numpy as np

import cv2

import argparse

import torch
from nvjpeg_torch import Jpeg




if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')


  args = parser.parse_args()

  jpeg = Jpeg()
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)

  yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
  
  yuv = torch.from_numpy(yuv)
  y = yuv[:image.shape[0]]

  uv = yuv[image.shape[0]:].view(2, image.shape[0]//2, image.shape[1]//2).contiguous()
    
  data = jpeg.encode_yuv(y, uv)

  filename = path.join("out", path.basename(args.filename))
  with open(filename, "wb") as f:
    f.write(data.numpy().tobytes())

