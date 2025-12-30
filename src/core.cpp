#include "miniTensor/core.h"
#include "miniTensor/functions.h"
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <typeinfo>
#include <map>
#include <iomanip>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

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

// Get function type name (demangle C++ type names)
std::string get_function_type_name(const Function* func) {
    if (!func) return "None";
    
    const char* name = typeid(*func).name();
    
    // Try to demangle (works on GCC/Clang)
    #ifdef __GNUC__
        int status = 0;
        char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        if (status == 0 && demangled) {
            std::string result(demangled);
            free(demangled);
            // Extract class name (remove namespace)
            size_t pos = result.find_last_of("::");
            if (pos != std::string::npos) {
                result = result.substr(pos + 1);
            }
            return result;
        }
    #endif
    
    // Fallback: return raw name or simplified version
    std::string result(name);
    // Remove common prefixes
    if (result.find("N11miniTensor") == 0) {
        // Try to extract class name from mangled name
        size_t pos = result.find_last_of("E");
        if (pos != std::string::npos && pos > 0) {
            // Look for class name before 'E'
            size_t start = pos - 1;
            while (start > 0 && std::isdigit(result[start])) start--;
            if (start < pos - 1) {
                result = result.substr(start + 1, pos - start - 1);
            }
        }
    }
    return result.empty() ? "Function" : result;
}

// Helper function to collect all nodes in computation graph
void collect_graph_nodes(const Tensor* tensor,
                        std::map<const Tensor*, int>& tensor_ids,
                        std::map<const Function*, int>& func_ids,
                        std::vector<const Tensor*>& tensors,
                        std::vector<const Function*>& functions,
                        int& next_tensor_id,
                        int& next_func_id) {
    if (!tensor) return;
    
    // Assign ID to tensor if not seen
    if (tensor_ids.find(tensor) == tensor_ids.end()) {
        tensor_ids[tensor] = next_tensor_id++;
        tensors.push_back(tensor);
    }
    
    // Process creator function
    if (tensor->creator) {
        auto func = tensor->creator.get();
        if (func_ids.find(func) == func_ids.end()) {
            func_ids[func] = next_func_id++;
            functions.push_back(func);
            
            // Recursively process inputs (these are shared_ptr, so safe to use)
            for (auto& input : func->inputs) {
                if (input) {
                    collect_graph_nodes(input.get(), tensor_ids, func_ids, tensors, functions, 
                                      next_tensor_id, next_func_id);
                }
            }
        }
    }
}

// Print computation graph in text format
void print_computation_graph(const std::shared_ptr<Tensor>& tensor, 
                            std::ostream& os, 
                            bool dot_format) {
    if (!tensor) {
        os << "Empty tensor\n";
        return;
    }
    
    if (dot_format) {
        os << computation_graph_to_dot(tensor);
        return;
    }
    
    std::map<const Tensor*, int> tensor_ids;
    std::map<const Function*, int> func_ids;
    std::vector<const Tensor*> tensors;
    std::vector<const Function*> functions;
    int next_tensor_id = 0;
    int next_func_id = 0;
    
    // Collect all nodes
    collect_graph_nodes(tensor.get(), tensor_ids, func_ids, tensors, functions, 
                       next_tensor_id, next_func_id);
    
    os << "=== Computation Graph ===\n\n";
    
    // Print tensors
    os << "Tensors:\n";
    for (const auto* t : tensors) {
        int id = tensor_ids[t];
        os << "  T" << id;
        if (!t->name.empty()) {
            os << " (" << t->name << ")";
        }
        auto shape = t->shape();
        os << ": shape=(" << shape.first << ", " << shape.second << ")";
        if (t->creator) {
            os << " <- " << get_function_type_name(t->creator.get());
        } else {
            os << " [leaf]";
        }
        os << "\n";
    }
    
    os << "\nFunctions:\n";
    for (const auto* f : functions) {
        int id = func_ids[f];
        os << "  F" << id << ": " << get_function_type_name(f);
        os << " (generation=" << f->generation << ")";
        os << "\n    Inputs: ";
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            if (f->inputs[i]) {
                os << "T" << tensor_ids[f->inputs[i].get()];
                if (i < f->inputs.size() - 1) os << ", ";
            }
        }
        os << "\n    Outputs: ";
        bool first = true;
        for (auto& output : f->outputs) {
            if (auto output_ptr = output.lock()) {
                if (!first) os << ", ";
                os << "T" << tensor_ids[output_ptr.get()];
                first = false;
            }
        }
        os << "\n";
    }
    
    os << "\nRoot Tensor: T" << tensor_ids[tensor.get()] << "\n";
}

// Overload for const Tensor*
void print_computation_graph(const Tensor* tensor, 
                            std::ostream& os, 
                            bool dot_format) {
    if (!tensor) {
        os << "Empty tensor\n";
        return;
    }
    
    if (dot_format) {
        // For const Tensor*, we can't easily get shared_ptr, so create a temporary
        // This is safe since we're only reading
        auto temp = std::shared_ptr<Tensor>(const_cast<Tensor*>(tensor), [](Tensor*){});
        os << computation_graph_to_dot(temp);
        return;
    }
    
    std::map<const Tensor*, int> tensor_ids;
    std::map<const Function*, int> func_ids;
    std::vector<const Tensor*> tensors;
    std::vector<const Function*> functions;
    int next_tensor_id = 0;
    int next_func_id = 0;
    
    // Collect all nodes
    collect_graph_nodes(tensor, tensor_ids, func_ids, tensors, functions, 
                       next_tensor_id, next_func_id);
    
    os << "=== Computation Graph ===\n\n";
    
    // Print tensors
    os << "Tensors:\n";
    for (const auto* t : tensors) {
        int id = tensor_ids[t];
        os << "  T" << id;
        if (!t->name.empty()) {
            os << " (" << t->name << ")";
        }
        auto shape = t->shape();
        os << ": shape=(" << shape.first << ", " << shape.second << ")";
        if (t->creator) {
            os << " <- " << get_function_type_name(t->creator.get());
        } else {
            os << " [leaf]";
        }
        os << "\n";
    }
    
    os << "\nFunctions:\n";
    for (const auto* f : functions) {
        int id = func_ids[f];
        os << "  F" << id << ": " << get_function_type_name(f);
        os << " (generation=" << f->generation << ")";
        os << "\n    Inputs: ";
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            if (f->inputs[i]) {
                os << "T" << tensor_ids[f->inputs[i].get()];
                if (i < f->inputs.size() - 1) os << ", ";
            }
        }
        os << "\n    Outputs: ";
        bool first = true;
        for (auto& output : f->outputs) {
            if (auto output_ptr = output.lock()) {
                if (!first) os << ", ";
                os << "T" << tensor_ids[output_ptr.get()];
                first = false;
            }
        }
        os << "\n";
    }
    
    os << "\nRoot Tensor: T" << tensor_ids[tensor] << "\n";
}

// Generate DOT format for graphviz
std::string computation_graph_to_dot(const std::shared_ptr<Tensor>& tensor) {
    if (!tensor) {
        return "digraph G { empty [label=\"Empty tensor\"]; }";
    }
    
    std::map<const Tensor*, int> tensor_ids;
    std::map<const Function*, int> func_ids;
    std::vector<const Tensor*> tensors;
    std::vector<const Function*> functions;
    int next_tensor_id = 0;
    int next_func_id = 0;
    
    // Collect all nodes
    collect_graph_nodes(tensor.get(), tensor_ids, func_ids, tensors, functions, 
                       next_tensor_id, next_func_id);
    
    std::ostringstream oss;
    oss << "digraph ComputationGraph {\n";
    oss << "  rankdir=LR;\n";
    oss << "  node [shape=box, style=rounded];\n\n";
    
    // Add tensor nodes
    for (const auto* t : tensors) {
        int id = tensor_ids[t];
        std::string label = "T" + std::to_string(id);
        if (!t->name.empty()) {
            label += "\\n" + t->name;
        }
        auto shape = t->shape();
        label += "\\nshape=(" + std::to_string(shape.first) + "," + std::to_string(shape.second) + ")";
        
        std::string color = t->creator ? "lightblue" : "lightgreen";
        oss << "  T" << id << " [label=\"" << label << "\", fillcolor=" << color 
            << ", style=\"rounded,filled\"];\n";
    }
    
    // Add function nodes
    for (const auto* f : functions) {
        int id = func_ids[f];
        std::string func_name = get_function_type_name(f);
        oss << "  F" << id << " [label=\"" << func_name << "\", shape=ellipse, "
            << "fillcolor=lightyellow, style=filled];\n";
    }
    
    oss << "\n";
    
    // Add edges: function -> output tensor
    for (const auto* f : functions) {
        int func_id = func_ids[f];
        for (auto& output : f->outputs) {
            if (auto output_ptr = output.lock()) {
                int tensor_id = tensor_ids[output_ptr.get()];
                oss << "  F" << func_id << " -> T" << tensor_id << ";\n";
            }
        }
    }
    
    // Add edges: input tensor -> function
    for (const auto* f : functions) {
        int func_id = func_ids[f];
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            if (f->inputs[i]) {
                int tensor_id = tensor_ids[f->inputs[i].get()];
                oss << "  T" << tensor_id << " -> F" << func_id;
                if (f->inputs.size() > 1) {
                    oss << " [label=\"in" << i << "\"]";
                }
                oss << ";\n";
            }
        }
    }
    
    oss << "}\n";
    return oss.str();
}

// Tensor methods for graph visualization
void Tensor::print_graph(std::ostream& os, bool dot_format) const {
    print_computation_graph(this, os, dot_format);
}

std::string Tensor::to_dot() const {
    // For DOT format, we need shared_ptr for the recursive collection
    // Create a temporary shared_ptr (safe since we're only reading)
    auto temp = std::shared_ptr<Tensor>(const_cast<Tensor*>(this), [](Tensor*){});
    return computation_graph_to_dot(temp);
}

} // namespace miniTensor

