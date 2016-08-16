#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

maxtree_extension = Extension('_maxtree',
                              sources=['maxtree.i','../cppsrc/maxtree.cpp'],
                              swig_opts=['-c++', '-I../cppsrc'],
                              include_dirs=['../cppsrc', numpy.get_include()],
                              extra_compile_args=['-std=c++11'],
                              extra_link_args=['-std=c++11'],
                              )

setup(name='MaxTree',
      version='1.0',
      description='Max Tree algorithm',
      author='Lionel Gueguen',
      author_email='lgueguen@uber.com',
      url='NA',
      ext_modules = [maxtree_extension],
      py_modules =['maxtree','component_tree','component_tree_test']
     )
