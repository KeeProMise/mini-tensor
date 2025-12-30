#ifndef MINITENSOR_MODELS_H
#define MINITENSOR_MODELS_H

#include "layers.h"
#include "functions.h"
#include <memory>
#include <vector>
#include <functional>

namespace miniTensor {

class Model : public Layer {
public:
    Model();
    virtual ~Model() = default;
};

class MLP : public Model {
private:
    std::vector<std::shared_ptr<Linear>> layers_;
    std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)> activation_;

public:
    MLP(const std::vector<int>& fc_output_sizes, 
        std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)> activation = sigmoid);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override;
    std::vector<std::shared_ptr<Parameter>> params() override;
};

} // namespace miniTensor

#endif // MINITENSOR_MODELS_H

