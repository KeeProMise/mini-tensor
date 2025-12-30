#ifndef MINITENSOR_LAYERS_H
#define MINITENSOR_LAYERS_H

#include "core.h"
#include "functions.h"
#include <memory>
#include <vector>
#include <unordered_set>
#include <string>

namespace miniTensor {

class Layer {
protected:
    std::unordered_set<std::string> params_;

public:
    Layer();
    virtual ~Layer() = default;
    
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    virtual std::vector<std::shared_ptr<Parameter>> params();
    void cleargrads();
    
    void register_param(const std::string& name, const std::shared_ptr<Parameter>& param);
};

class Linear : public Layer {
private:
    int in_size_;
    int out_size_;
    std::shared_ptr<Parameter> W_;
    std::shared_ptr<Parameter> b_;
    bool nobias_;

public:
    Linear(int out_size, bool nobias = false, int in_size = -1);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override;
    std::vector<std::shared_ptr<Parameter>> params() override;
    
private:
    void init_W();
};

class TwoLayerNet : public Layer {
private:
    std::shared_ptr<Linear> l1_;
    std::shared_ptr<Linear> l2_;

public:
    TwoLayerNet(int hidden_size, int out_size);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override;
    std::vector<std::shared_ptr<Parameter>> params() override;
};

} // namespace miniTensor

#endif // MINITENSOR_LAYERS_H

