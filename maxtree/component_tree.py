"""
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
"""

import maxtree
import numpy as np

class MaxTree(object):
    '''
    compute max tree object from a 2d image array
    '''


    def __init__(self, im):
        '''
        construct the max tree for an image
        image shall be either uint8, uint16 or uint32
        '''
        self.layer_count=0
        self.featsDescription={}
        self._compute(im)
        self.nb_connected_components = self.mt.getNbCC()
        self.feats = np.zeros((self.nb_connected_components,0), np.float32)
    
    def __str__(self):
        st = "MaxTree of an image of size: "+ str(self.mt.getHeight())+"x"+str(self.mt.getWidth())+"\n"
        st+= "\t nb connected components: "+ str(self.nb_connected_components)+"\n"
        st+= "\t self.featsDescription.keys(): "+ str(self.featsDescription.keys())
        return st
    
    
        
    def _compute(self,im):
        if len(im.shape)!=2:
            raise IndexError('input to MaxTree should a 2d array, len(im.shape)!=2')
        if not(im.dtype == np.uint16):
            raise TypeError('input to MaxTree should be of type np.uint16')
        
        temp_im = np.ascontiguousarray(im)
        self.mt = maxtree.MT(temp_im)
        self.mt.compute()
    
    def parent(self,cc_idx):
        '''
        retrieve the index of the parent connected component
        input cc_idx: an int32 representing the index of a connected component
        return: an int32 representing the index of the parent connected component
        '''
        return self.mt.getParent(cc_idx)
    
    def contrast(self, cc_idx):
        '''
        retrieve the contrast of the current connected component with its the parent connected component
        input cc_idx: an int32 representing the index of a connected component
        return: a float32 representing the contrast with its parent
        '''
        return self.mt.getDiff(cc_idx)

    def _log_scaling(self, feats, log_scaling):
        '''
        feats come as 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'angle', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7'
        we modify these to 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7', 'lshape', 'cosangle', 'sinangle'
        '''
        lshape = np.sqrt(feats[:,7])/ (np.sqrt(feats[:,6])+log_scaling)
        cosangle = np.cos(feats[:,5])
        sinangle = np.sin(feats[:,5])
        feats = np.delete(feats, 5, 1)
        for j in range(4,feats.shape[1]):
            feats[:,j] = np.log(np.abs(feats[:,j]) + log_scaling)

        feats = np.concatenate((feats,lshape[:,None],cosangle[:,None],sinangle[:,None]), axis=1)
        return feats


    def compute_shape_attributes(self, log_scaling=1e-10):
        '''
        computes the shape attributes in the maxtree. The class object gets appended the attributes in self.feats
        input log_scaling: if None, no log scaling, other wise apply log scaling
        return: nothing
        note: attributes computed are 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7', 'lshape', 'cosangle', 'sinangle'
        '''
        start_i = self.feats.shape[1]
        feats = self.mt.computeShapeAttributes_swig()
        feats = self._log_scaling(feats, log_scaling)
        self.feats = np.concatenate((self.feats,feats), axis=1)
        featsD=['xmin', 'ymin', 'xmax', 'ymax', 'area', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7', 'lshape', 'cosangle', 'sinangle']
        featsDescription = {f:i+start_i for i,f in enumerate(featsD)}
        self.featsDescription.update(featsDescription)
    
    def compute_layer_attributes(self, layers):
        '''
        computes the layer attributes. The class object gets appended the attributes in self.feats
        input layers: a numpy 2d or 3d array, of size w x h or w x h x nb_bands
        return: nothing
        note: attributes computed are 'average_xx', 'std_xx', 'min_val_xx', 'max_val_xx', where xx is self.layer_count
        '''
        s = layers.shape
        s+=(1,)
        if len(s)<3:
            raise IndexError('input to layer attributes should a 2d or 3 d array')
        if s[0]!=self.mt.getHeight() or s[1]!= self.mt.getWidth():
            raise IndexError('input to layer attributes should a 2d or 3 d array of same dimensions as input image')
        
        featsD=['average', 'std', 'min_val', 'max_val']
        if s[2]==1:
            loc_lyr = np.float32(layers).reshape((s[0],s[1]))
            start_i = self.feats.shape[1]
            feats = self.mt.computeLayerAttributes_swig(loc_lyr)
            self.feats = np.concatenate((self.feats,feats), axis=1)
            featsDescription={f+'_'+str(self.layer_count):i+start_i for i,f in enumerate(featsD)}
            self.featsDescription.update(featsDescription)
            self.layer_count+=1
        else:
            for i in range(s[2]):
                self.compute_layer_attributes(layers[:,:,i])
            
    
    def getAttributes(self,keys=[]):
        '''
        retrieve the attributes matrix depending on an attribute key or more than one key
        input keys: either a string or list of string
        return: a np.float32 array of size nbcc x nbattributes
        note: valid keys are given by self.featsDescription.keys()
        '''
        if len(keys)==0:
            return self.feats
        elif type(keys) is str and keys in self.featsDescription.keys():
            return self.getAttributes([keys]).squeeze()
        elif type(keys) is list:
            for k in keys:
                if k not in self.featsDescription.keys():
                    raise KeyError(k+" is not part of valid keys in self.featsDescription")
            tmp_idx = [self.featsDescription[f] for f in keys]
            return self.feats[:,tmp_idx]
        else:
            raise KeyError(keys+" is not part of valid keys in self.featsDescription")
        
    def filter(self, idx, scores = None):
        '''
        filter the max tree in direct mode, by stacking the connected components which are retained
        input idx: np.uint32 array of the indices of connected components which are retained
        input scores (optional): np.float32 array providing a score the previous indices. 
        return: a 2d array of the filtered image
        note: example of indices computation, idx = (self.getAttributes('area')>1000).nonzero()[0]
        '''
        if scores is None:
            idx_tmp = np.uint32(idx)
            return self.mt.filter_swig(idx_tmp)
        else:
            idx_tmp = np.uint32(idx)
            s_tmp = np.float32(scores)
            return self.mt.filter_swig(idx_tmp,s_tmp)

    def filterAttributes(self, idx, scores):
        '''
        filter the max tree in direct mode, by stacking the connected components scores passed
        input idx: np.uint32 array of the indices of connected components which are retained
        input scores: np.float32 array providing a score the previous indices.
        return: a 3d array of the attribute images, corresponding to count, avg, std, min, max
        note: example of indices computation, idx = (self.getAttributes('area')>1000).nonzero()[0]
                                              scores = self.getAttributes('area')[idx]
        '''
        idx_tmp = np.uint32(idx)
        s_tmp = np.float32(scores)
        out = self.mt.computePerPixelAttributes_swig(idx_tmp, s_tmp)
        o = out.reshape((5,out.shape[0]/5,out.shape[1])).transpose((1,2,0))
        return o
        