from setuptools import setup, Extension
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


assert os.path.exists('/usr/src/jetson_multimedia_api'), "Jetson multimedia API not found"


setup(
    name='nvjpeg_torch',
    version="0.1.1",
    packages=['nvjpeg_torch'],

    ext_modules=[
        CUDAExtension('nvjpeg_cuda', 
          [ 'nvjpeg_cuda.cpp', 
            '/usr/src/jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp',
            '/usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp',
            '/usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp',

            '/usr/src/jetson_multimedia_api/samples/common/classes/NvBufSurface.cpp',
            '/usr/src/jetson_multimedia_api/argus/samples/utils/CUDAHelper.cpp'
        ],
        include_dirs=['/usr/src/jetson_multimedia_api/argus/samples/utils', '/usr/src/jetson_multimedia_api/include', '/usr/src/jetson_multimedia_api/include/libjpeg-8b'], 

        define_macros=[('JPEGCODER_ARCH', 'jetson')],
        library_dirs=['/usr/lib/aarch64-linux-gnu/tegra', 'build/lib'],
        libraries=['cudart', 'nvjpeg', 'cuda', 'pthread', 'nvv4l2', 'nvbufsurface', 'nvosd', 'nvbuf_utils', 'nvbufsurftransform'])
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
