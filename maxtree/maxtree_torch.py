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
import torch
from torch.autograd import Function
import _maxtreetorch as maxtreetorch

def _log_scaling(feats):
    """
    feats come as 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'angle', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7'
    we modify these to 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7', 'lshape', 'cosangle', 'sinangle'
    """
    epsilon = 1e-10
    lshape = torch.sqrt(feats[:, 7:8]) / (torch.sqrt(feats[:, 6:7]) + epsilon)
    cosangle = torch.cos(feats[:, 5:6])
    sinangle = torch.sin(feats[:, 5:6])
    for j in range(5, feats.shape[1]):
        feats[:, j] = torch.log(torch.abs(feats[:, j]) + epsilon)

    feats = torch.cat((feats[:,:5], feats[:,6:], lshape, cosangle, sinangle), dim=1)
    return feats

class DifferentialMaxtreeFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        ctx: context
        input: image of size (H, W). The image gets converted to int16 internally. Make sure your values are well scaled.
        weight: float tensor of size (17,1), representing the linear combination to attributes
        bias: float tensor of size (1,), representing the bias
        returns:
            a float tensor of size (H, W), which is the filtering of the input with scores meeting the gaussian criterion
            expressed with the mean and inv_diagonal_cov.
        """
        short_input = input.to(torch.int16).to(torch.device("cpu")).contiguous()
        maxtree_parent, maxtree_diff, attributes = maxtreetorch.maxtree(short_input)
        rescaled_attributes = _log_scaling(attributes)


        # sigmoid(attributes * W + B)
        linear_attr = rescaled_attributes.mm(weight) + bias[None, :]
        cc_scores = torch.sigmoid(linear_attr)[:, 0]

        ctx.save_for_backward(maxtree_parent, maxtree_diff, cc_scores, rescaled_attributes)
        filtered_output = maxtreetorch.forward(maxtree_parent, maxtree_diff, cc_scores.float())[0]

        return filtered_output

    @staticmethod
    def backward(ctx, grad_filtered):
        """
        ctx: context
        grad_filtered: gradient of loss with output, delta_L / delta_output of size (H,W)
        returns:
            delta_L / delta_input: float tensor of shape (H, W)
            delta_L / delta_weight: float tensor of shape (17,1)
            delat_L / delta_bias: float tensor of shape (1,)
        """
        maxtree_parent, maxtree_diff, cc_scores, rescaled_attributes = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        _grad_input, _grad_cc_score = maxtreetorch.backward(maxtree_parent, maxtree_diff, grad_filtered.contiguous())

        sigmoid_derivative = cc_scores * (1 - cc_scores)
        if ctx.needs_input_grad[0]:
            grad_input = _grad_input
        if ctx.needs_input_grad[1]:
            grad_weight = (_grad_cc_score[:, None] * sigmoid_derivative[:, None] * rescaled_attributes).sum(dim=0)[:, None]
        if ctx.needs_input_grad[2]:
            grad_bias = (_grad_cc_score * sigmoid_derivative).sum()[None]

        return grad_input, grad_weight, grad_bias


class DifferentialMaxtree(torch.nn.Module):
    NUM_FEATURES = 17

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(self.NUM_FEATURES, 1))
        self.bias = torch.nn.Parameter(torch.empty(1))
        self.initialize()

    def initialize(self):
            self.weight.data.uniform_(-0.01, 0.01)
            self.bias.data.fill_(0)

    def forward(self, input):
        """
        input: tensor of shape (H,W). The tensor gets converted to int16 internally.
        returns:
            tensor of shape (H,W)
        """
        return DifferentialMaxtreeFunction.apply(input, self.weight, self.bias)
