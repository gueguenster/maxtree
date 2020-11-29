#!/usr/bin/env python

# Get numpy and pytorch first
MIN_NUMPY_VERSION = '1.8.0'
MIN_TORCH_VERSION = '1.2.0'

from setuptools import dist

dist.Distribution().fetch_build_eggs(['torch>={}'.format(MIN_TORCH_VERSION),
                                      'numpy>={}'.format(MIN_NUMPY_VERSION)])

# Build the extensions, and installs
import os
from setuptools import setup, Extension
import numpy
from torch.utils import cpp_extension
from maxtree import __version__

MAXTREE_CPP_SRC = os.path.join(os.path.dirname(__file__), 'cppsrc')
maxtree_extension = Extension('_maxtree',
                              sources=['./maxtree/maxtree_wrap.cpp', './cppsrc/maxtree.cpp'],
                              include_dirs=[MAXTREE_CPP_SRC, numpy.get_include()],
                              )

torch_extension = cpp_extension.CppExtension('_maxtreetorch',
                                             ['./cppsrc/maxtree.cpp', './cppsrc/torch.cpp'],
                                             extra_compile_args=['-fopenmp'])

setup(name='maxtree',
      version=__version__,
      description='Max Tree algorithm',
      author='Lionel Gueguen',
      author_email='gueguenster@gmail.com',
      url='https://github.com/gueguenster/maxtree',
      ext_modules=[maxtree_extension, torch_extension],
      packages=['maxtree'],
      include_package_data=True,
      python_requires='>=3.6',
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      },
      install_requires=[
          "numpy>={}".format(MIN_NUMPY_VERSION),
          "torch>={}".format(MIN_TORCH_VERSION),
      ],
      )
