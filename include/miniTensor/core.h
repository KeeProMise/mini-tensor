#ifndef MINITENSOR_CORE_H
#define MINITENSOR_CORE_H

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <map>
#include <Eigen/Dense>

namespace miniTensor {

using Array = Eigen::MatrixXf;
using ArrayRef = Eigen::Ref<const Eigen::MatrixXf>;
using ArrayMap = Eigen::Map<const Eigen::MatrixXf>;

// Forward declarations
class Tensor;
class Function;

// Config for backpropagation
class Config {
public:
    static bool enable_backprop;
};

// Convert scalar to array
Array as_array(float x);
Array as_array(const Array& x);

// Tensor class - represents a tensor with gradient
class Tensor {
public:
    Array data;
    std::string name;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<Function> creator;  // Use shared_ptr to keep Function alive
    int generation;

    Tensor(const Array& data, const std::string& name = "");
    Tensor(float scalar, const std::string& name = "");
    
    void set_creator(std::shared_ptr<Function> func);
    void backward(bool retain_grad = false, bool create_graph = false);
    void cleargrad();
    
    // Shape properties
    std::pair<int, int> shape() const;
    int ndim() const;
    int size() const;
    
    // Note: Operators are implemented via free functions in functions.h
    
    std::string repr() const;
    std::string toString() const;  // Java-style toString method
    
    // Computation graph visualization
    void print_graph(std::ostream& os = std::cout, bool dot_format = false) const;
    std::string to_dot() const;
};

// Stream output operator (like std::cout << tensor)
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Tensor>& tensor);

// Function base class for automatic differentiation
class Function : public std::enable_shared_from_this<Function> {
public:
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::weak_ptr<Tensor>> outputs;
    int generation;

    Function();
    virtual ~Function() = default;
    
    std::shared_ptr<Tensor> call(const std::vector<std::shared_ptr<Tensor>>& inputs);
    virtual std::vector<Array> forward(const std::vector<Array>& xs) = 0;
    virtual std::vector<Array> backward(const std::vector<Array>& gys) = 0;
};

// Helper functions
std::shared_ptr<Tensor> as_tensor(const std::shared_ptr<Tensor>& obj);
std::shared_ptr<Tensor> as_tensor(const Array& obj);
std::shared_ptr<Tensor> as_tensor(float scalar);

// Computation graph visualization
std::string get_function_type_name(const Function* func);
void print_computation_graph(const std::shared_ptr<Tensor>& tensor, 
                            std::ostream& os = std::cout, 
                            bool dot_format = false);
void print_computation_graph(const Tensor* tensor, 
                            std::ostream& os = std::cout, 
                            bool dot_format = false);
std::string computation_graph_to_dot(const std::shared_ptr<Tensor>& tensor);

// Context manager for config
class ConfigContext {
    std::string name;
    bool old_value;
public:
    ConfigContext(const std::string& name, bool value);
    ~ConfigContext();
};

ConfigContext no_grad();

// Parameter class (alias for Tensor)
class Parameter : public Tensor {
public:
    Parameter(const Array& data, const std::string& name = "");
    Parameter(float scalar, const std::string& name = "");
};

} // namespace miniTensor

#endif // MINITENSOR_CORE_H

