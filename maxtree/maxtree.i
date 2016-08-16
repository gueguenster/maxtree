/*
COPYRIGHT

All contributions by L. Gueguen:
Copyright (c) 2016
All rights reserved.


LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
%module maxtree

%{
#define SWIG_FILE_WITH_INIT
#include "maxtree.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%numpy_typemaps(unsigned short, NPY_USHORT, unsigned int)
%numpy_typemaps(float, NPY_FLOAT, unsigned int)
%numpy_typemaps(unsigned int, NPY_UINT, unsigned int)
%numpy_typemaps(float, NPY_FLOAT, unsigned int)

%apply (unsigned short* IN_ARRAY2, unsigned int DIM1, unsigned int DIM2) {(unsigned short* imarray, unsigned int width, unsigned int height)};
%apply (float* IN_ARRAY2, unsigned int DIM1, unsigned int DIM2) {(float* imarray, unsigned int width, unsigned int height)};
%apply (unsigned int* IN_ARRAY1, unsigned int DIM1) {(unsigned int * retained, unsigned int lr)}
%apply (float* IN_ARRAY1, unsigned int DIM1) {(float * score, unsigned int ls)}
%apply (float** ARGOUTVIEWM_ARRAY2, unsigned int * DIM1, unsigned int * DIM2) {(float** outf, unsigned int * wf, unsigned int * hf)}
%apply (unsigned short** ARGOUTVIEWM_ARRAY2, unsigned int * DIM1, unsigned int * DIM2) {(unsigned short** out, unsigned int * w, unsigned int * h)}
%apply (unsigned int** ARGOUTVIEWM_ARRAY1, unsigned int * DIM1) {(unsigned int ** par, unsigned int * l)}


%include "maxtree.h"


%extend MaxTree< unsigned short > {
%pythoncode %{
    def __getstate__(self):
        args={}
        args['bytes'] = self.serialize_swig()
        return args
    def __setstate__(self, state):
        dummy = int(0)
        self.__init__(dummy, state['bytes'])
%}
}
%template(MT) MaxTree< unsigned short >;





