#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

maxtree_extension = Extension('_maxtree',
                              sources=['./maxtree/maxtree_wrap.cpp','./cppsrc/maxtree.cpp'],
                              include_dirs=['./cppsrc', numpy.get_include()],
                              extra_compile_args=['-std=c++11'],
                              extra_link_args=['-std=c++11'],
                              )

setup(name='MaxTree',
      version='0.1.0',
      description='Max Tree algorithm',
      author='Lionel Gueguen',
      author_email='gueguenster@gmail.com',
      url='https://github.com/gueguenster/maxtree',
      ext_modules = [maxtree_extension],
      py_modules =['maxtree.maxtree','maxtree.component_tree']
     )
