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

import unittest
from component_tree import MaxTree
import numpy as np
import pickle

class Test(unittest.TestCase):


    def setUp(self):
        self.img_us = np.uint16(np.random.rand(123,456)*1024)
        self.img_uc = np.uint16(np.random.rand(56,34)*255)
        pass


    def tearDown(self):
        pass


    def testType(self):
        mt = MaxTree(self.img_uc)
        self.assertIsNotNone(mt, "could not create max tree on uint8")
        mt = MaxTree(self.img_us)
        self.assertIsNotNone(mt, "could not create max tree on uint16")
        
        img = np.uint32(np.random.rand(356,234)*1024)
        with self.assertRaises(TypeError):
            mt = MaxTree(img)
            
        img = np.uint16(np.random.rand(356,234,3)*1024)
        with self.assertRaises(IndexError):
            mt = MaxTree(img)

    
    def testLayerAttributes(self):
        mt = MaxTree(self.img_uc)
        
        w,h = self.img_uc.shape
        lyr = np.float32(np.random.rand(w,h))
        mt.compute_layer_attributes(lyr)
        
        self.assertEqual(mt.feats.shape[1], 4, "layer attributes are not sufficient")
        
        layers = np.float32(np.random.rand(w,h,3))
        mt.compute_layer_attributes(layers)
        self.assertEqual(mt.feats.shape[1], 4 + 3*4, "layer attributes are not sufficient")
    
    def testShapeAttributes(self):
        mt = MaxTree(self.img_us)
        mt.compute_shape_attributes()
        
        self.assertEqual(mt.feats.shape[1], 17, "layer attributes are not sufficient")
    
    def testGetAttributes(self):
        mt = MaxTree(self.img_us)
        mt.compute_shape_attributes()
        
        areas = mt.getAttributes('area')
        self.assertEqual(len(areas.shape),1, "getAttributes should return one dimension")
        self.assertEqual(areas.shape[0], mt.mt.getNbCC(), "the areas should be as long as the number of CC")
        
        morethana = mt.getAttributes(['area','xmin'])
        self.assertEqual(len(morethana.shape),2, "getAttributes should return more than one dimension")
        self.assertEqual(morethana.shape[1],2, "getAttributes should return more than one dimension")
        
        with self.assertRaises(KeyError):
            badkey = mt.getAttributes(['area','notakey'])
            
        with self.assertRaises(KeyError):
            badkey = mt.getAttributes('notakey')
    
    def testFilter(self):
        mt = MaxTree(self.img_us)
        mt.compute_shape_attributes()
        
        idx = (mt.getAttributes('area')>100).nonzero()[0]
        out = mt.filter(idx)
        self.assertEqual(out.shape[0], mt.mt.getHeight(), "output of filtering is not of right dimensions")
        self.assertEqual(out.shape[1], mt.mt.getWidth(), "output of filtering is not of right dimensions")
        
        idx = (mt.getAttributes('area')>-1).nonzero()[0]
        out = mt.filter(idx)
        self.assertTrue((out==self.img_us).all(), "output should be equal to input image")
        
        scores = np.ones(idx.shape,np.float32)
        out = mt.filter(idx,scores)
        self.assertTrue((out==self.img_us).all(), "output should be equal to input image")
    
    def testParentDiff(self):
        mt = MaxTree(self.img_us)
        p = mt.parent(0)
        self.assertGreaterEqual(p, 0, "could retrieve parent")
        pp = mt.parent(mt.nb_connected_components-1)
        self.assertGreaterEqual(pp, 0, "could retrieve parent")
            
        d = mt.contrast(0)
        self.assertGreater(d, 0, "contrast should greater than 0")
    
    def testStr(self):
        mt = MaxTree(self.img_us)
        mt.compute_shape_attributes()
        print mt
    
    def testPickle(self):
        mt = MaxTree(self.img_uc)
        #non working for the moment
        #mt.compute_shape_attributes()
        
        #bytes = pickle.dumps(mt)
        #mtt = pickle.loads(bytes)
        
        #idx = np.asarray([0])
        
        #print mtt.mt.getNbCC()
        #print mt.mt.getNbCC()
        #self.assertEqual(mtt.mt.getNbCC(), mt.mt.getNbCC(), "pickling did not work")
        #self.assertEqual((mtt.feats == mt.feats).all(), "pickling did not work")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test']
    unittest.main()