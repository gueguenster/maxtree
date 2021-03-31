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

NUM_FEATURES = 17

def _log_scaling(feats):
    """
    feats come as 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'angle', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7'
    we modify these to 'xmin', 'ymin', 'xmax', 'ymax', 'area', 'pca_big', 'pca_small', 'hu_1', 'hu_2','hu_3','hu_4','hu_5','hu_6', 'hu_7', 'lshape', 'cosangle', 'sinangle'
    """
    epsilon = 1e-10
    lshape = torch.sqrt(feats[:, 7:8]) / (torch.sqrt(feats[:, 6:7]) + epsilon)
    cosangle = torch.cos(feats[:, 5:6])
    sinangle = torch.sin(feats[:, 5:6])
    feats[:, 4] = torch.log(feats[:, 4])
    feats[:, 6:] = torch.log(torch.abs(feats[:, 6:]) + epsilon) * torch.sign(feats[:, 6:])
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
        maxtree_parent, maxtree_diff, maxtree_cc2hdr, attributes = maxtreetorch.maxtree(short_input)
        rescaled_attributes = _log_scaling(attributes)

        # sigmoid(attributes * W + B)
        linear_attr = rescaled_attributes.mm(cpu_weight) + cpu_bias[None, :]
        cc_scores = torch.sigmoid(linear_attr)[:, 0]

        ctx.save_for_backward(maxtree_parent, maxtree_diff, maxtree_cc2hdr, cc_scores, rescaled_attributes)
        filtered_output = maxtreetorch.forward(maxtree_parent, maxtree_diff, maxtree_cc2hdr, cc_scores.float())[0]

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
        maxtree_parent, maxtree_diff, maxtree_cc2hdr, cc_scores, rescaled_attributes = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        cpu_grad_filtered = grad_filtered.to(torch.device("cpu")).contiguous()
        _grad_input, _grad_cc_score = maxtreetorch.backward(maxtree_parent, maxtree_diff, maxtree_cc2hdr,
                                                            cpu_grad_filtered)

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

        # extract maxtrees
        maxtrees = []
        outputs = []
        for single_channel, w, b in zip(short_input, cpu_weight, cpu_bias):
            mt_par, mt_diff, mt_cc2ph, mt_attr = maxtreetorch.maxtree(single_channel)
            rescaled_attributes = _log_scaling(mt_attr)

            linear_attr = rescaled_attributes.mm(w) + b[None, :]
            cc_scores = torch.sigmoid(linear_attr)[:, 0]
            maxtrees += [mt_par, mt_diff, mt_cc2ph, cc_scores, rescaled_attributes]

            filtered_output = maxtreetorch.forward(mt_par, mt_diff, mt_cc2ph, cc_scores.float())[0]
            outputs.append(filtered_output.to(input.device))

        ctx.save_for_backward(*maxtrees)
        return torch.stack(outputs)

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

        cpu_grad_filtered = grad_filtered.to(torch.device("cpu")).contiguous()
        n = cpu_grad_filtered.shape[0]

        grad_inputs = []
        grad_weights = []
        grad_biass = []

        for idx in range(n):
            maxtree_parent = ctx.saved_tensors[idx * 5 + 0]
            maxtree_diff = ctx.saved_tensors[idx * 5 + 1]
            maxtree_cc2hdr = ctx.saved_tensors[idx * 5 + 2]
            cc_scores = ctx.saved_tensors[idx * 5 + 3]
            rescaled_attributes = ctx.saved_tensors[idx * 5 + 4]

            local_grad_filtered = cpu_grad_filtered[idx]
            _grad_input, _grad_cc_score = maxtreetorch.backward(maxtree_parent, maxtree_diff, maxtree_cc2hdr,
                                                                local_grad_filtered)

            sigmoid_derivative = cc_scores * (1 - cc_scores)
            if ctx.needs_input_grad[0]:
                grad_input = _grad_input.to(grad_filtered.device)
                grad_inputs.append(grad_input)

            if ctx.needs_input_grad[1]:
                _grad_weight = (_grad_cc_score[:, None] * sigmoid_derivative[:, None] * rescaled_attributes).sum(dim=0)[:,
                               None]
                grad_weight = _grad_weight.to(grad_filtered.device)
                grad_weights.append(grad_weight)

            if ctx.needs_input_grad[2]:
                _grad_bias = (_grad_cc_score * sigmoid_derivative).sum()[None]
                grad_bias = _grad_bias.to(grad_filtered.device)
                grad_biass.append(grad_bias)

        if len(grad_inputs) > 0:
            grad_inputs = torch.stack(grad_inputs)
        else:
            grad_inputs = None

        if len(grad_weights) > 0:
            grad_weights = torch.stack(grad_weights)
        else:
            grad_weights = None

        if len(grad_biass) > 0:
            grad_biass = torch.stack(grad_biass)
        else:
            grad_biass = None

        return grad_inputs, grad_weights, grad_biass


class SingleChannelDifferentialMaxtree(torch.nn.Module):
    """
    SingleChannelDifferentialMaxtree is a torch layer which represents a Maxtree/mintree parameterized filter. The
    filter learns a mapping from the tree connected components to [0,1]. These scores are then applied to each
    connected components, resulting in the filtered output. The mapping is implemented as a logits on the shape
    properties of each connected component.
    Example:
        import torch
        from maxtree.maxtree_torch import SingleChannelDifferentialMaxtree

        mt_layer = DifferentialMaxtree(kind='mintree')
        mt_layer.initialize()

        h, w = 64, 64
        input = torch.randint(-10, 32, (h, w))
        output = mt_layer(input)
    """

    def __init__(self,
                 kind: str = 'maxtree',
                 scaling: float = 1.0):
        """
        Instantiate a single channel differential maxtree layer. The layer is a torch.nn.Module and implements a
        forward operator. The layer has internal parameters: weights of size (17, 1), and bias of size (1,).
        These parameters define the filtering applied to the input image.
        :param kind: a string in the set {'maxtree', 'mintree'}. When 'maxtree', the peaks of the input image are
            analyzed. When 'mintree', the valleys of the input are considered. In practice, when 'mintree' is defined,
            the input is inverted by multiplying it by -1, and the output is inverted back.
        :param scaling: a scaling factor applied to the input prior to be analyzed by the maxtree algorithm. As the
            maxtree works on torch.int16, the input is first scaled, then quantized and moved onto cpu internally.
            The maxtree algorithm works on torch.int16(input * scaling).to(torch.device("cpu")). The output is
            scaled back, and moved to the input type and device.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(NUM_FEATURES, 1))
        self.bias = torch.nn.Parameter(torch.empty(1))

        if kind not in ['maxtree', 'mintree']:
            raise ValueError("The kind of maxtrees must be in ['maxtree', 'mintree'].")
        self.kind = kind

        if scaling <= 0:
            raise ValueError("The scaling factor is strictly positive.")
        self.scaling = scaling
        if self.kind == 'mintree':
            self.scaling *= -1.0

    def initialize(self):
        """
        Initialize the layer weights.
        """
        self.weight.data.uniform_(-0.01, 0.01)
        self.bias.data.fill_(0)

    def forward(self, input):
        """
        Runs the differential maxtree filters on the inputs.
        :param input: tensor of shape (H,W).
            The tensor gets multiplied by the scaling factor internally, and is casted to torch.int16 on the cpu.
            The scaling factor allows to not loose information, prior to the quantization to int16. The output
            is inversely scaled by the scaling factor, and casted to the input type and device.
        returns:
            tensor of shape (H,W). The results of applying the differential maxtree filtering on the image.
            The output type and device match the input type and device.
        """
        return DifferentialMaxtreeFunction.apply(input, self.weight, self.bias)


class DifferentialMaxtree(torch.nn.Module):
    """
    DifferentialMaxtree is a torch layer which represents multiple Maxtree/mintree parameterized filters. The
    filters learn a mapping from the tree connected components to [0,1]. These scores are then applied to each
    connected components, resulting in the filtered outputs. The mapping is implemented as a logits on the shape
    properties of each connected component.
    Example:
        import torch
        from maxtree.maxtree_torch import DifferentialMaxtree

        b, N, h, w = 2, 10, 64, 64

        mt_layer = DifferentialMaxtree(num_channels=N, scaling=10.)
        mt_layer.initialize()

        input = torch.randint(-10, 32, (b, N, h, w))
        output = mt_layer(input)
    """
    def __init__(self,
                 num_channels: int = 1,
                 kind: str = 'maxtree',
                 scaling: float = 1.0):
        """
        Instantiate a differential maxtree layer. The layer is a torch.nn.Module and implements a forward operator.
        The layer has internal parameters: weights of size (N, 17, 1), and bias of size (N, 1). These parameters define
        the filtering applied to the input images.
        :param num_channels: the number of channels N that the input must have. The input size to this layer is expected
            to be (B, N, H, W), and the output of this layer is the same.
        :param kind: a string in the set {'maxtree', 'mintree'}. When 'maxtree', the peaks of the input images are
            analyzed. When 'mintree', the valeys of the input are considered. In practice, when 'mintree' is defined,
            the input is inverted by multiplying it by -1, and the output is inverted back.
        :param scaling: a scaling factor applied to the input prior to be analyzed by the maxtree algorithm. As the
            maxtree works on torch.int16, the input is first scaled, then quantized and moved onto cpu internally.
            The maxtree algorithm works on torch.int16(input * scaling).to(torch.device("cpu")). The output is
            scaled back, and moved to the input type and device.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_channels, NUM_FEATURES, 1))
        self.bias = torch.nn.Parameter(torch.empty(num_channels, 1))

        if kind not in ['maxtree', 'mintree']:
            raise ValueError("The kind of maxtrees must be in ['maxtree', 'mintree'].")
        self.kind = kind

        if scaling <= 0:
            raise ValueError("The scaling factor is strictly positive.")
        self.scaling = scaling
        if self.kind == 'mintree':
            self.scaling *= -1.0

    def initialize(self):
        """
        Initialize the layer weights.
        """
        self.weight.data.uniform_(-0.01, 0.01)
        self.bias.data.fill_(0)

    def forward(self, batched_input):
        """
        Runs the differential maxtree filters on the inputs.
        :param batched_input: tensor of shape (B,N,H,W).
            The tensor gets multiplied by the scaling factor internally, and is casted to torch.int16 on the cpu.
            The scaling factor allows to not loose information, prior to the quantization to int16. The output
            is inversely scaled by the scaling factor, and casted to the input type and device.
        returns:
            tensor of shape (B,N,H,W). The results of applying the differential maxtree filtering on each channel.
            The output type and device match the input type and device.
        """
        scaled_input = batched_input * self.scaling
        batched_out = []
        for input in scaled_input:
            filtered = ManyDifferentialMaxtreeFunction.apply(input, self.weight, self.bias)
            batched_out.append(filtered)
        batched_out = torch.stack(batched_out, dim=0) / self.scaling
        return batched_out
