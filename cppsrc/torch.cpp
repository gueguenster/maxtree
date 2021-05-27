/*
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
*/
#include <torch/extension.h>
#include <vector>
#include <map>

#include "maxtree.h"

std::vector<at::Tensor> maxtree(torch::Tensor channel) {
  auto width = (unsigned int) channel.size(0);
  auto height = (unsigned int) channel.size(1);

  //Maxtree
  auto maxtree = MaxTree<short>(channel.data_ptr<short>(), width, height );
  maxtree.compute();

  std::vector<int64_t> dims = {width, height};

  //Parent
  vector<int> parent(begin(maxtree.getParent()), end(maxtree.getParent()));   //check that values are not overflowing
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  auto mt_parent = torch::tensor(parent, opts).reshape(dims);

  //CC2PixelHeaders
  vector<int> cc2ph(begin(maxtree.getCC2PixelHeader()), end(maxtree.getCC2PixelHeader()));
  auto mt_cc2ph = torch::tensor(cc2ph, opts);

  //Diff
  opts = torch::TensorOptions().dtype(torch::kInt16);
  vector<short> diff(begin(maxtree.getDiff()), end(maxtree.getDiff()));
  auto mt_diff = torch::tensor(diff, opts).reshape(dims);

  //Attributes
  vector < vector < double > > _attributes = maxtree.computeShapeAttributes( );
  ui nbcc = maxtree.getNbCC ();
  ui kDim = 15; //
  vector < double > attributes;
  for(auto vec: _attributes){
    for(auto v: vec){
        attributes.push_back(v);
    }
  }
  auto opts_attr = torch::TensorOptions().dtype(torch::kFloat64);
  std::vector<int64_t> attr_dims = {nbcc, kDim};
  auto _mt_attributes = torch::tensor(attributes, opts_attr).reshape(attr_dims);
  auto mt_attributes = _mt_attributes.to(torch::kFloat32);

  return {mt_parent, mt_diff, mt_cc2ph, mt_attributes};

}

std::vector<torch::Tensor> maxtree_forward(torch::Tensor mt_parent, torch::Tensor mt_diff,
    torch::Tensor mt_cc2ph,
    torch::Tensor cc_scores) {
  // Reinstantiate the maxtree
  ui width = (ui) mt_parent.size(0);
  ui height = (ui) mt_parent.size(1);
  ui nbCCs = (ui) mt_cc2ph.size(0);
  auto _parent =  mt_parent.data_ptr<int>();
  auto _cc2ph = mt_cc2ph.data_ptr<int>();
  auto _diff = mt_diff.data_ptr<short>();
  const vector<ui> parent(_parent, _parent + width*height);
  const vector<ui> cc2ph(_cc2ph, _cc2ph + nbCCs);
  const vector<short> diff(_diff, _diff + width*height);
  auto maxtree = MaxTree<short>(parent, diff, cc2ph, width, height);

  // Convert the scores
  vector < pair<ui, float> > cc_score_vector;
  auto acc = cc_scores.accessor<float, 1>();
  pair<ui, float> pair;

  for (int64_t i = 0; i <  cc_scores.size(0); i++) {
    if(acc[i] != 0.){
        pair.first = (ui) i;
        pair.second = acc[i];
        cc_score_vector.push_back(pair);
    }
  }

  // Filter
  auto _filtered_channel = maxtree.filter(cc_score_vector);
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<int64_t> dims = {width, height};
  auto filtered_channel = torch::tensor(_filtered_channel, opts).reshape(dims);

  return {filtered_channel};
}

std::vector<torch::Tensor> maxtree_backward(torch::Tensor mt_parent, torch::Tensor mt_diff,
    torch::Tensor mt_cc2ph,
    torch::Tensor grad_filtered) {

    // Reinstantiate the maxtree
    ui width = (ui) mt_parent.size(0);
    ui height = (ui) mt_parent.size(1);
    ui nbCCs = (ui) mt_cc2ph.size(0);
    auto _parent =  mt_parent.data_ptr<int>();
    auto _diff = mt_diff.data_ptr<short>();
    auto _cc2ph = mt_cc2ph.data_ptr<int>();
    const vector<ui> parent(_parent, _parent + width*height);
    const vector<ui> cc2ph(_cc2ph, _cc2ph + nbCCs);
    const vector<short> diff(_diff, _diff + width*height);
    auto maxtree = MaxTree<short>(parent, diff, cc2ph, width, height);

    // CC score and input gradient
    float * accessor = grad_filtered.data_ptr<float>();
    vector <float> _grad_filtered(accessor, accessor + maxtree.getNbpixels());
    vector < vector < double > > _attributes = maxtree.computeLayerAttributes(_grad_filtered);
    ui nbcc = maxtree.getNbCC ();
    vector < float > grad_cc_scores;
    vector < pair<ui, float> > cc_delta_score_vector;
    for(ui cc_idx=0; cc_idx< nbcc; cc_idx++){
        float avg_grad_cc = (float) _attributes[cc_idx][0];
        // TODO: check if multiplying with area is relevant
        float delta = (float) maxtree.getDiff(cc_idx);
        grad_cc_scores.push_back(avg_grad_cc * delta);

        pair<ui, float> pair;
        pair.first = cc_idx;
        pair.second = avg_grad_cc;
        cc_delta_score_vector.push_back(pair);
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto _grad_input = maxtree.filter(cc_delta_score_vector);

    std::vector<int64_t> dims = {width, height};
    auto grad_input = torch::tensor(_grad_input, opts).reshape(dims);
    auto gradient_cc_scores = torch::tensor(grad_cc_scores, opts);

    return {grad_input, gradient_cc_scores};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maxtree", &maxtree, "Maxtree compute");
  m.def("forward", &maxtree_forward, "Maxtree forward");
  m.def("backward", &maxtree_backward, "Maxtree backward");
}