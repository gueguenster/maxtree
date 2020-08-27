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
import _maxtreetorch as maxtreetorch
from torch.autograd import Function


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

    feats = torch.cat((feats[:, :5], feats[:, 6:], lshape, cosangle, sinangle), dim=1)
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
        # convert inputs to be on cpu
        short_input = input.to(torch.int16).to(torch.device("cpu")).contiguous()
        cpu_weight = weight.to(torch.device("cpu"))
        cpu_bias = bias.to(torch.device("cpu"))

        # extract maxtree
        maxtree_parent, maxtree_diff, attributes = maxtreetorch.maxtree(short_input)
        rescaled_attributes = _log_scaling(attributes)

        # sigmoid(attributes * W + B)
        linear_attr = rescaled_attributes.mm(cpu_weight) + cpu_bias[None, :]
        cc_scores = torch.sigmoid(linear_attr)[:, 0]

        ctx.save_for_backward(maxtree_parent, maxtree_diff, cc_scores, rescaled_attributes)
        filtered_output = maxtreetorch.forward(maxtree_parent, maxtree_diff, cc_scores.float())[0]

        # reset the maxtree output on the input device
        return filtered_output.to(input.device)

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

        cpu_grad_filtered = grad_filtered.to(torch.device("cpu")).contiguous()
        _grad_input, _grad_cc_score = maxtreetorch.backward(maxtree_parent, maxtree_diff, cpu_grad_filtered)

        sigmoid_derivative = cc_scores * (1 - cc_scores)
        if ctx.needs_input_grad[0]:
            grad_input = _grad_input.to(grad_filtered.device)
        if ctx.needs_input_grad[1]:
            _grad_weight = (_grad_cc_score[:, None] * sigmoid_derivative[:, None] * rescaled_attributes).sum(dim=0)[:,
                           None]
            grad_weight = _grad_weight.to(grad_filtered.device)
        if ctx.needs_input_grad[2]:
            _grad_bias = (_grad_cc_score * sigmoid_derivative).sum()[None]
            grad_bias = _grad_bias.to(grad_filtered.device)

        return grad_input, grad_weight, grad_bias


class ManyDifferentialMaxtreeFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        ctx: context
        input: image of size (N, H, W). The image gets converted to int16 internally. Make sure your values are well scaled.
        weight: float tensor of size (N, 17, 1), representing the linear combination to attributes
        bias: float tensor of size (N, 1), representing the bias
        returns:
            a float tensor of size (N, H, W), which is the filtering of the input with scores meeting the gaussian criterion
            expressed with the mean and inv_diagonal_cov.
        """
        # convert inputs to be on cpu
        short_input = input.to(torch.int16).to(torch.device("cpu")).contiguous()
        cpu_weight = weight.to(torch.device("cpu"))
        cpu_bias = bias.to(torch.device("cpu"))

        # extract maxtree
        all = maxtreetorch.maxtrees(short_input)
        mt_parents = []
        mt_diffs = []
        mt_rescaled_attributes = []
        mt_cc_scores = []
        for i in range(len(all) // 3):
            mt_parents.append(all[i * 3])
            mt_diffs.append(all[i * 3 + 1])
            rescaled_attributes = _log_scaling(all[i * 3 + 2])
            mt_rescaled_attributes.append(rescaled_attributes)

            # sigmoid(attributes * W + B)
            linear_attr = rescaled_attributes.mm(cpu_weight[i]) + cpu_bias[i][None, :]
            cc_scores = torch.sigmoid(linear_attr)[:, 0]
            mt_cc_scores.append(cc_scores)

        tensors_to_save = mt_parents + mt_diffs + mt_cc_scores + mt_rescaled_attributes
        ctx.save_for_backward(*tensors_to_save)
        reorg_tensors = []
        for p, d, cc in zip(mt_parents, mt_diffs, mt_cc_scores):
            reorg_tensors.append(p)
            reorg_tensors.append(d)
            reorg_tensors.append(cc.float())
        filtered_outputs = maxtreetorch.many_forward(reorg_tensors)

        # reset the maxtree output on the input device
        return torch.stack(filtered_outputs).to(input.device)

    @staticmethod
    def backward(ctx, grad_filtered):
        """
        ctx: context
        grad_filtered: gradient of loss with output, delta_L / delta_output of size (N, H,W)
        returns:
            delta_L / delta_input: float tensor of shape (N, H, W)
            delta_L / delta_weight: float tensor of shape (N, 17, 1)
            delat_L / delta_bias: float tensor of shape (N, 1)
        """
        saved_tensors = ctx.saved_tensors
        n = len(saved_tensors) // 4
        mt_parents = saved_tensors[:n]
        mt_diffs = saved_tensors[n:(2*n)]
        mt_cc_scores = saved_tensors[(2*n):(3*n)]
        mt_rescaled_attributes = saved_tensors[(3*n):(4*n)]
        grad_input = grad_weight = grad_bias = None

        cpu_grad_filtered = grad_filtered.to(torch.device("cpu")).contiguous()
        grads_filters = [f[0] for f in torch.split(cpu_grad_filtered, 1)]
        reorg_tensors = []
        for p, d, f in zip(mt_parents, mt_diffs, grads_filters):
            reorg_tensors.append(p)
            reorg_tensors.append(d)
            reorg_tensors.append(f)
        grad_all = maxtreetorch.many_backward(reorg_tensors)

        many_grad_inputs = []
        many_grad_cc_scores = []
        for i in range(len(grad_all) // 2):
            many_grad_inputs.append(grad_all[i * 2].to(grad_filtered.device))
            many_grad_cc_scores.append(grad_all[i * 2 + 1].to(grad_filtered.device))

        sigmoid_derivatives = []
        for cc_scores in mt_cc_scores:
            cc_scores = cc_scores.to(grad_filtered.device)
            sigmoid_derivative = cc_scores * (1 - cc_scores)
            sigmoid_derivatives.append(sigmoid_derivative)

        if ctx.needs_input_grad[0]:
            grad_input = torch.stack(many_grad_inputs)
        if ctx.needs_input_grad[1]:
            all_weights = []
            for _grad_cc_score, sigmoid_derivative, rescaled_attributes in zip(many_grad_cc_scores, sigmoid_derivatives,
                                                                               mt_rescaled_attributes):
                rescaled_attributes = rescaled_attributes.to(grad_filtered.device)
                _grad_weight = (_grad_cc_score[:, None] * sigmoid_derivative[:, None] * rescaled_attributes).sum(dim=0)[
                               :, None]
                all_weights.append(_grad_weight)
            grad_weight = torch.stack(all_weights)
        if ctx.needs_input_grad[2]:
            all_bias = []
            for _grad_cc_score, sigmoid_derivative in zip(many_grad_cc_scores, sigmoid_derivatives):
                _grad_bias = (_grad_cc_score * sigmoid_derivative).sum()[None]
                all_bias.append(_grad_bias)

            grad_bias = torch.stack(all_bias)

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


class CDifferentialMaxtree(torch.nn.Module):
    NUM_FEATURES = 17

    def __init__(self, num_channels: int = 1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_channels, self.NUM_FEATURES, 1))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, 1))
        self.initialize()

    def initialize(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.bias.data.fill_(0)

    def forward(self, batched_input):
        """
        input: tensor of shape (B,N,H,W). The tensor gets converted to int16 internally.
        returns:
            tensor of shape (B,N,H,W)
        """
        batched_out = []
        for input in batched_input:
            filtered = ManyDifferentialMaxtreeFunction.apply(input, self.weight, self.bias)
            batched_out.append(filtered)
        return torch.stack(batched_out, dim=0)
