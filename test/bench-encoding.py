from turbojpeg import TurboJPEG
from nvjpeg_torch import Jpeg
import torch

import cv2
import time

from functools import partial


from threading import Thread
from queue import Queue

import gc

import argparse

class Timer:     
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start



class NvJpeg(object):
  def __init__(self):
    self.jpeg = Jpeg()

  def encode(self, image, quality=94):
    # image = image.cuda()
    compressed = self.jpeg.encode_yuv_420(image, quality)


class Threaded(object):
  def __init__(self, create_jpeg, size=8):
        # Image file writers
    self.queue = Queue(size)
    self.threads = [Thread(target=self.encode_thread, args=()) 
        for _ in range(size)]

    self.create_jpeg = create_jpeg

    
    for t in self.threads:
        t.start()


  def encode_thread(self):
    jpeg = self.create_jpeg()
    item = self.queue.get()
    while item is not None:
      image, quality = item

      result = jpeg.encode(image, quality)
      item = self.queue.get()


  def encode(self, image, quality=94):
    self.queue.put((image, quality))


  def stop(self):
      for _ in self.threads:
          self.queue.put(None)

      for t in self.threads:
        t.join()
      



def bench_threaded(create_encoder, images, threads, warmup=100):
  threads = Threaded(create_encoder, threads)

  for image in images[:warmup]:
    threads.encode(image)

  with Timer() as t:
    for image in images:
      threads.encode(image)

    threads.stop()
    # torch.cuda.synchronize()

  return len(images) / t.interval


def bench_encoder(create_encoder, images, warmup=20):
  encoder = create_encoder()

  for image in images[:warmup]:
    encoder.encode(image)

  with Timer() as t:
    for image in images:
      encoder.encode(image)

  return len(images) / t.interval


def main(args):
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)
  yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)


  num_threads = args.j


  yuv_images = [torch.from_numpy(yuv_image)] * args.n
  print(f'nvjpeg (on cpu): {bench_threaded(NvJpeg, yuv_images, 4):>5.1f} images/s')

  images = [image] * args.n
  print(f'turbojpeg threaded j={num_threads}: {bench_threaded(TurboJPEG, images, num_threads):>5.1f} images/s')


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')

  parser.add_argument('-j', default=8, type=int, help='run multi-threaded')
  parser.add_argument('-n', default=400, type=int, help='number of images to encode')

  args = parser.parse_args()
  main(args)
  gc.collect()

  # main(args)
  

