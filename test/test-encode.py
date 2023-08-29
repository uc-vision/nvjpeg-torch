
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

  data = jpeg.encode_yuv_420(torch.from_numpy(yuv))

  filename = path.join("out", path.basename(args.filename))
  with open(filename, "wb") as f:
    f.write(data.numpy().tobytes())

