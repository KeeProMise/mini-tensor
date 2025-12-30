#ifndef MINITENSOR_OPTIMIZERS_H
#define MINITENSOR_OPTIMIZERS_H

#include "models.h"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>

namespace miniTensor {

class Optimizer {
protected:
    std::shared_ptr<Model> target_;
    std::vector<std::function<void(std::vector<std::shared_ptr<Parameter>>&)>> hooks_;

public:
    Optimizer();
    virtual ~Optimizer() = default;
    
    Optimizer& setup(std::shared_ptr<Model> target);
    void update();
    void add_hook(std::function<void(std::vector<std::shared_ptr<Parameter>>&)> f);
    
protected:
    virtual void update_one(std::shared_ptr<Parameter> param) = 0;
};

class SGD : public Optimizer {
private:
    float lr_;

public:
    SGD(float lr = 0.01f);
    void update_one(std::shared_ptr<Parameter> param) override;
};

class MomentumSGD : public Optimizer {
private:
    float lr_;
    float momentum_;
    std::unordered_map<size_t, Array> vs_;

public:
    MomentumSGD(float lr = 0.01f, float momentum = 0.9f);
    void update_one(std::shared_ptr<Parameter> param) override;
};

} // namespace miniTensor

#endif // MINITENSOR_OPTIMIZERS_H

