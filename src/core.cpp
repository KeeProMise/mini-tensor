#include "miniTensor/core.h"
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace miniTensor {

// Initialize static member
bool Config::enable_backprop = true;

// Convert scalar to array
Array as_array(float x) {
    Array arr(1, 1);
    arr(0, 0) = x;
    return arr;
}

Array as_array(const Array& x) {
    return x;
}

// Tensor implementation
Tensor::Tensor(const Array& data, const std::string& name)
    : data(data), name(name), grad(nullptr), creator(nullptr), generation(0) {
}

Tensor::Tensor(float scalar, const std::string& name)
    : data(as_array(scalar)), name(name), grad(nullptr), creator(nullptr), generation(0) {
}

void Tensor::set_creator(std::shared_ptr<Function> func) {
    creator = func;
    generation = func->generation + 1;
}

void Tensor::backward(bool retain_grad, bool create_graph) {
    if (!Config::enable_backprop) {
        return;
    }
    
    if (grad == nullptr) {
        Array ones = Array::Ones(data.rows(), data.cols());
        grad = std::make_shared<Tensor>(ones);
    }

    std::vector<std::shared_ptr<Function>> funcs;
    std::unordered_set<Function*> seen_set;

    auto add_func = [&](std::shared_ptr<Function> f) {
        if (f && seen_set.find(f.get()) == seen_set.end()) {
            funcs.push_back(f);
            seen_set.insert(f.get());
            std::sort(funcs.begin(), funcs.end(), 
                [](const std::shared_ptr<Function>& a, const std::shared_ptr<Function>& b) { 
                    return a->generation < b->generation; 
                });
        }
    };

    if (creator != nullptr) {
        add_func(creator);
    } else {
        // If no creator, this is a leaf variable, nothing to do
        return;
    }

    while (!funcs.empty()) {
        std::shared_ptr<Function> f = funcs.back();
        funcs.pop_back();

        std::vector<Array> gys;
        bool skip_this_function = false;
        for (auto& output : f->outputs) {
            if (auto output_ptr = output.lock()) {
                if (output_ptr->grad) {
                    gys.push_back(output_ptr->grad->data);
                } else {
                    skip_this_function = true;
                    break;
                }
            } else {
                // Output was deleted, skip this function
                skip_this_function = true;
                break;
            }
        }

        // Skip this function if any output has None grad or was deleted
        if (skip_this_function || gys.empty()) {
            continue;
        }

        bool old_enable = Config::enable_backprop;
        Config::enable_backprop = create_graph;
        
        std::vector<Array> gxs = f->backward(gys);
        Config::enable_backprop = old_enable;

        for (size_t i = 0; i < f->inputs.size() && i < gxs.size(); ++i) {
            auto& x = f->inputs[i];
            if (x->grad == nullptr) {
                x->grad = std::make_shared<Tensor>(gxs[i]);
            } else {
                x->grad->data += gxs[i];
            }

            if (x->creator != nullptr) {
                add_func(x->creator);
            }
        }

        if (!retain_grad) {
            for (auto& output : f->outputs) {
                if (auto output_ptr = output.lock()) {
                    // Only clear if this output is not the final output (the one we called backward on)
                    // Actually, Python clears all intermediate grads, so we should too
                    output_ptr->grad = nullptr;
                }
            }
        }
    }
}

void Tensor::cleargrad() {
    grad = nullptr;
}

std::pair<int, int> Tensor::shape() const {
    return {data.rows(), data.cols()};
}

int Tensor::ndim() const {
    return 2; // Eigen Matrix is always 2D
}

int Tensor::size() const {
    return data.size();
}

// Helper function to format a float value in Python style
std::string format_float(float val) {
    std::ostringstream oss;
    oss.precision(6);
    oss << std::fixed << val;
    std::string str = oss.str();
    // Remove trailing zeros after decimal point
    size_t pos = str.find_last_not_of('0');
    if (pos != std::string::npos && str[pos] == '.') {
        pos++;  // Keep one zero after decimal
    }
    if (pos != std::string::npos) {
        str.erase(pos + 1);
    }
    return str;
}

// Helper function to format array in Python style
std::string format_array_python_style(const Array& arr) {
    std::ostringstream oss;
    int rows = arr.rows();
    int cols = arr.cols();
    
    if (rows == 0 || cols == 0) {
        oss << "[]";
        return oss.str();
    }
    
    // For 1x1 scalar
    if (rows == 1 && cols == 1) {
        oss << format_float(arr(0, 0));
        return oss.str();
    }
    
    // For row vector (1xN)
    if (rows == 1) {
        oss << "[";
        for (int j = 0; j < cols; ++j) {
            oss << format_float(arr(0, j));
            if (j < cols - 1) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }
    
    // For column vector (Nx1)
    if (cols == 1) {
        oss << "[";
        for (int i = 0; i < rows; ++i) {
            oss << "\n        [" << format_float(arr(i, 0)) << "]";
            if (i < rows - 1) oss << ",";
        }
        oss << "\n    ]";
        return oss.str();
    }
    
    // For 2D matrix
    oss << "[";
    for (int i = 0; i < rows; ++i) {
        oss << "\n        [";
        for (int j = 0; j < cols; ++j) {
            oss << format_float(arr(i, j));
            if (j < cols - 1) oss << ", ";
        }
        oss << "]";
        if (i < rows - 1) oss << ",";
    }
    oss << "\n    ]";
    return oss.str();
}

std::string Tensor::repr() const {
    std::ostringstream oss;
    
    if (data.size() == 0) {
        oss << "tensor([])";
    } else {
        oss << "tensor(" << format_array_python_style(data) << ")";
    }
    
    // 如果有 grad，也输出 grad
    if (grad != nullptr) {
        oss << ", grad=tensor(" << format_array_python_style(grad->data) << ")";
    }
    
    return oss.str();
}

std::string Tensor::toString() const {
    return repr();  // Java-style toString, delegates to repr()
}

// Stream output operator implementation
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.toString();
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor) {
    if (tensor) {
        os << tensor->toString();
    } else {
        os << "nullptr";
    }
    return os;
}

// Function implementation
Function::Function() : generation(0) {
}

std::shared_ptr<Tensor> Function::call(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    this->inputs = inputs;
    
    std::vector<Array> xs;
    for (auto& input : inputs) {
        xs.push_back(input->data);
    }

    std::vector<Array> ys = forward(xs);
    
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (auto& y : ys) {
        outputs.push_back(std::make_shared<Tensor>(y));
    }

    if (Config::enable_backprop) {
        int max_gen = 0;
        for (auto& input : inputs) {
            max_gen = std::max(max_gen, input->generation);
        }
        this->generation = max_gen;

        // Store this Function as shared_ptr so it stays alive
        // Use enable_shared_from_this to get shared_ptr to this
        std::shared_ptr<Function> self_ptr = shared_from_this();
        for (auto& output : outputs) {
            output->set_creator(self_ptr);
            this->outputs.push_back(std::weak_ptr<Tensor>(output));
        }
    }

    return outputs[0];
}

// Helper functions
std::shared_ptr<Tensor> as_tensor(const std::shared_ptr<Tensor>& obj) {
    return obj;
}

std::shared_ptr<Tensor> as_tensor(const Array& obj) {
    return std::make_shared<Tensor>(obj);
}

std::shared_ptr<Tensor> as_tensor(float scalar) {
    return std::make_shared<Tensor>(scalar);
}

// ConfigContext implementation
ConfigContext::ConfigContext(const std::string& name, bool value) : name(name) {
    if (name == "enable_backprop") {
        old_value = Config::enable_backprop;
        Config::enable_backprop = value;
    }
}

ConfigContext::~ConfigContext() {
    if (name == "enable_backprop") {
        Config::enable_backprop = old_value;
    }
}

ConfigContext no_grad() {
    return ConfigContext("enable_backprop", false);
}

// Parameter implementation
Parameter::Parameter(const Array& data, const std::string& name)
    : Tensor(data, name) {
}

Parameter::Parameter(float scalar, const std::string& name)
    : Tensor(scalar, name) {
}

} // namespace miniTensor

