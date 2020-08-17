#!/usr/bin/env python

from setuptools import setup, Extension
import numpy
from torch.utils import cpp_extension
from maxtree import __version__

maxtree_extension = Extension('_maxtree',
                              sources=['./maxtree/maxtree_wrap.cpp','./cppsrc/maxtree.cpp'],
                              include_dirs=['./cppsrc', numpy.get_include()],
                              extra_compile_args=['-std=c++11'],
                              extra_link_args=['-std=c++11'],
                              )

torch_extension = cpp_extension.CppExtension('_maxtreetorch',
                                             ['./cppsrc/maxtree.cpp', './cppsrc/torch.cpp'])


setup(name='maxtree',
      version=__version__,
      description='Max Tree algorithm',
      author='Lionel Gueguen',
      author_email='gueguenster@gmail.com',
      url='https://github.com/gueguenster/maxtree',
      ext_modules = [maxtree_extension, torch_extension],
      packages =['maxtree'],
      include_package_data=True,
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      }
     )
