"""
COPYRIGHT

All contributions by L. Gueguen:
Copyright (c) 2020
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


import numpy as np
import torch
import _maxtreetorch as maxtreetorch

from maxtree.component_tree import MaxTree
from maxtree.maxtree_torch import CDifferentialMaxtree, ManyDifferentialMaxtreeFunction


class MaxtreeTorchTest(unittest.TestCase):
    def setUp(self):
        self.img_short_np = np.int16(np.random.rand(120, 50) * 1024)
        self.img_ushort_np = np.uint16(self.img_short_np)
        self.img_short_torch = torch.tensor(self.img_short_np)
        pass

    # def test_construction(self):
    #     # construction
    #     mt = MaxTree(self.img_ushort_np)
    #     self.assertIsNotNone(mt, "could not create max tree on uint16")
    #     mt.compute_shape_attributes()
    #     attributes = mt.getAttributes()
    #
    #     mt_parent, mt_diff, mt_attributes = maxtreetorch.maxtree(self.img_short_torch)
    #     self.assertEqual(attributes.shape[0], mt_attributes.shape[0])
    #     self.assertLessEqual(torch.norm(torch.tensor(attributes[:, 0]) - mt_attributes[:, 0]), 1e-7)
    #     self.assertLessEqual(torch.norm(torch.tensor(attributes[:, 1]) - mt_attributes[:, 1]), 1e-7)
    #
    #     # filter
    #     idx = (attributes[:, 0] > -1).nonzero()[0]
    #     scores = np.ones(idx.shape, np.float32)
    #     out = mt.filter(idx, scores)
    #
    #     cc_scores = torch.ones_like(mt_attributes[:, 0])
    #     t_out = maxtreetorch.forward(mt_parent, mt_diff, cc_scores)[0]
    #     self.assertLessEqual(torch.norm(torch.tensor(out) - t_out), 1e-7)
    #
    #     # backward
    #     grad = torch.rand(self.img_short_torch.shape) - 0.5
    #     grad_input, grad_cc_scores = maxtreetorch.backward(mt_parent, mt_diff, grad)
    #     self.assertEqual(grad_input.shape, self.img_short_torch.shape)
    #     self.assertEqual(grad_cc_scores.shape, cc_scores.shape)
    #
    def test_function(self):
        nc = 256
        reference = torch.rand(1, nc, 100, 136) * 25
        reference[..., 5:10, 5:10] = 1
        lr = .01

        model = CDifferentialMaxtree(num_channels=nc)
        import time

        model.initialize()
        if True:
            model.cuda()
            reference = reference.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        previous_loss = None
        for i in range(50):
            t0 = time.time()

            optimizer.zero_grad()
            filtered = model(reference)
            loss = torch.nn.MSELoss()(filtered, reference)
            loss.backward()
            print(time.time() - t0)
            optimizer.step()
            if previous_loss is not None:
                self.assertLessEqual(loss, previous_loss)
                previous_loss = loss

if __name__ == "__main__":
    unittest.main()
