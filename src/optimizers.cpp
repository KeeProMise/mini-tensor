#include "miniTensor/optimizers.h"
#include <unordered_map>
#include <cmath>

namespace miniTensor {

// Optimizer implementation
Optimizer::Optimizer() : target_(nullptr) {
}

Optimizer& Optimizer::setup(std::shared_ptr<Model> target) {
    target_ = target;
    return *this;
}

void Optimizer::update() {
    if (!target_) return;
    
    std::vector<std::shared_ptr<Parameter>> params;
    for (auto& param : target_->params()) {
        if (param->grad) {
            params.push_back(param);
        }
    }
    
    // Preprocessing hooks
    for (auto& hook : hooks_) {
        hook(params);
    }
    
    // Update parameters
    for (auto& param : params) {
        update_one(param);
    }
}

void Optimizer::add_hook(std::function<void(std::vector<std::shared_ptr<Parameter>>&)> f) {
    hooks_.push_back(f);
}

// SGD implementation
SGD::SGD(float lr) : lr_(lr) {
}

void SGD::update_one(std::shared_ptr<Parameter> param) {
    if (param->grad) {
        // 检查梯度是否为NaN或Inf
        bool has_nan = false;
        bool has_inf = false;
        for (int i = 0; i < param->grad->data.rows(); ++i) {
            for (int j = 0; j < param->grad->data.cols(); ++j) {
                float val = param->grad->data(i, j);
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
            }
        }
        if (!has_nan && !has_inf) {
            param->data -= lr_ * param->grad->data;
        }
    }
}

// MomentumSGD implementation
MomentumSGD::MomentumSGD(float lr, float momentum) 
    : lr_(lr), momentum_(momentum) {
}

void MomentumSGD::update_one(std::shared_ptr<Parameter> param) {
    size_t param_id = reinterpret_cast<size_t>(param.get());
    
    if (vs_.find(param_id) == vs_.end()) {
        vs_[param_id] = Array::Zero(param->data.rows(), param->data.cols());
    }
    
    Array& v = vs_[param_id];
    v = momentum_ * v - lr_ * param->grad->data;
    param->data += v;
}

} // namespace miniTensor

