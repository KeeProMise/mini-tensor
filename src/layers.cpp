#include "miniTensor/layers.h"
#include "miniTensor/functions.h"
#include <random>
#include <cmath>

namespace miniTensor {

// Layer implementation
Layer::Layer() {
}

std::shared_ptr<Tensor> Layer::forward(const std::shared_ptr<Tensor>& x) {
    return x;
}

std::vector<std::shared_ptr<Parameter>> Layer::params() {
    std::vector<std::shared_ptr<Parameter>> result;
    for (const auto& name : params_) {
        // This would need reflection or a registry to work properly
        // Simplified version
    }
    return result;
}

void Layer::cleargrads() {
    for (auto& param : params()) {
        param->cleargrad();
    }
}

void Layer::register_param(const std::string& name, const std::shared_ptr<Parameter>& param) {
    params_.insert(name);
}

// Linear implementation
Linear::Linear(int out_size, bool nobias, int in_size)
    : out_size_(out_size), in_size_(in_size), nobias_(nobias) {
    if (in_size_ > 0) {
        init_W();
    } else {
        Array empty;
        W_ = std::make_shared<Parameter>(empty, "W");
    }
    
    if (!nobias_) {
        Array b_data = Array::Zero(1, out_size);
        b_ = std::make_shared<Parameter>(b_data, "b");
    } else {
        b_ = nullptr;
    }
}

void Linear::init_W() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, std::sqrt(1.0f / in_size_));
    
    Array W_data(in_size_, out_size_);
    for (int i = 0; i < in_size_; ++i) {
        for (int j = 0; j < out_size_; ++j) {
            W_data(i, j) = dis(gen);
        }
    }
    W_->data = W_data;
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& x) {
    if (W_->data.size() == 0) {
        in_size_ = x->data.cols();
        init_W();
    }
    
    if (b_) {
        return linear_func(x, std::make_shared<Tensor>(W_->data), std::make_shared<Tensor>(b_->data));
    } else {
        return linear_func(x, std::make_shared<Tensor>(W_->data), nullptr);
    }
}

std::vector<std::shared_ptr<Parameter>> Linear::params() {
    std::vector<std::shared_ptr<Parameter>> result;
    result.push_back(W_);
    if (b_) {
        result.push_back(b_);
    }
    return result;
}

// TwoLayerNet implementation
TwoLayerNet::TwoLayerNet(int hidden_size, int out_size) {
    l1_ = std::make_shared<Linear>(hidden_size);
    l2_ = std::make_shared<Linear>(out_size);
}

std::shared_ptr<Tensor> TwoLayerNet::forward(const std::shared_ptr<Tensor>& x) {
    auto y = sigmoid(l1_->forward(x));
    y = l2_->forward(y);
    return y;
}

std::vector<std::shared_ptr<Parameter>> TwoLayerNet::params() {
    std::vector<std::shared_ptr<Parameter>> result;
    auto l1_params = l1_->params();
    auto l2_params = l2_->params();
    result.insert(result.end(), l1_params.begin(), l1_params.end());
    result.insert(result.end(), l2_params.begin(), l2_params.end());
    return result;
}

} // namespace miniTensor

