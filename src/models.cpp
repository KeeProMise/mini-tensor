#include "miniTensor/models.h"
#include "miniTensor/functions.h"

namespace miniTensor {

// Model implementation
Model::Model() : Layer() {
}

// MLP implementation
MLP::MLP(const std::vector<int>& fc_output_sizes, 
         std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)> activation)
    : activation_(activation) {
    for (size_t i = 0; i < fc_output_sizes.size(); ++i) {
        layers_.push_back(std::make_shared<Linear>(fc_output_sizes[i]));
    }
}

std::shared_ptr<Tensor> MLP::forward(const std::shared_ptr<Tensor>& x) {
    auto y = x;
    for (size_t i = 0; i < layers_.size() - 1; ++i) {
        y = activation_(layers_[i]->forward(y));
    }
    return layers_.back()->forward(y);
}

std::vector<std::shared_ptr<Parameter>> MLP::params() {
    std::vector<std::shared_ptr<Parameter>> result;
    for (auto& layer : layers_) {
        auto layer_params = layer->params();
        result.insert(result.end(), layer_params.begin(), layer_params.end());
    }
    return result;
}

} // namespace miniTensor

