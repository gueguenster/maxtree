
#include <torch/extension.h>
#include <vector>

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

  return {mt_parent, mt_diff, mt_attributes};

}

std::vector<torch::Tensor> maxtree_forward(torch::Tensor mt_parent, torch::Tensor mt_diff, torch::Tensor cc_scores) {
  // Reinstantiate the maxtree
  ui width = (ui) mt_parent.size(0);
  ui height = (ui) mt_parent.size(1);
  auto _parent =  mt_parent.data_ptr<int>();
  auto _diff = mt_diff.data_ptr<short>();
  const vector<ui> parent(_parent, _parent + width*height);
  const vector<short> diff(_diff, _diff + width*height);
  auto maxtree = MaxTree<short>(parent, diff, width, height);

  // Convert the scores
  vector < pair<ui, float> > cc_score_vector;
  auto acc = cc_scores.accessor<float, 1>();
  for (int64_t i = 0; i <  cc_scores.size(0); i++) {
    pair<ui, float> pair;
    pair.first = (ui) i;
    pair.second = acc[i];
    cc_score_vector.push_back(pair);
  }

  // Filter
  auto _filtered_channel = maxtree.filter(cc_score_vector);
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<int64_t> dims = {width, height};
  auto filtered_channel = torch::tensor(_filtered_channel, opts).reshape(dims);

  return {filtered_channel};
}

std::vector<torch::Tensor> maxtree_backward(torch::Tensor mt_parent, torch::Tensor mt_diff, torch::Tensor grad_filtered) {
    // Reinstantiate the maxtree
    ui width = (ui) mt_parent.size(0);
    ui height = (ui) mt_parent.size(1);
    auto _parent =  mt_parent.data_ptr<int>();
    auto _diff = mt_diff.data_ptr<short>();
    const vector<ui> parent(_parent, _parent + width*height);
    const vector<short> diff(_diff, _diff + width*height);
    auto maxtree = MaxTree<short>(parent, diff, width, height);

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