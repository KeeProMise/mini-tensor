#ifndef MINITENSOR_FUNCTIONS_H
#define MINITENSOR_FUNCTIONS_H

#include "core.h"
#include <memory>
#include <vector>

namespace miniTensor {

// Forward declarations
class Add;
class Mul;
class Neg;
class Sub;
class Div;
class Pow;
class Exp;
class Sin;
class Cos;
class Tanh;
class Reshape;
class Transpose;
class Sum;
class BroadcastTo;
class SumTo;
class MatMul;
class MeanSquaredError;
class Log;
class LinearFunction;
class Sigmoid;
class GetItem;

// Function implementations
class Add : public Function {
public:
    std::pair<int, int> x0_shape, x1_shape;
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Mul : public Function {
public:
    std::pair<int, int> x0_shape, x1_shape;
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Neg : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Sub : public Function {
public:
    std::pair<int, int> x0_shape, x1_shape;
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Div : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Pow : public Function {
public:
    float c;
    Pow(float c) : c(c) {}
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Exp : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Sin : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Cos : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Tanh : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Reshape : public Function {
public:
    std::pair<int, int> shape;
    std::pair<int, int> x_shape;
    Reshape(int rows, int cols) : shape(rows, cols) {}
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Transpose : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Sum : public Function {
public:
    int axis;
    bool keepdims;
    std::pair<int, int> x_shape;
    Sum(int axis = -1, bool keepdims = false) : axis(axis), keepdims(keepdims) {}
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class BroadcastTo : public Function {
public:
    std::pair<int, int> shape;
    std::pair<int, int> x_shape;
    BroadcastTo(int rows, int cols) : shape(rows, cols) {}
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class SumTo : public Function {
public:
    std::pair<int, int> shape;
    std::pair<int, int> x_shape;
    SumTo(int rows, int cols) : shape(rows, cols) {}
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class MatMul : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class MeanSquaredError : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Log : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class LinearFunction : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

class Sigmoid : public Function {
public:
    std::vector<Array> forward(const std::vector<Array>& xs) override;
    std::vector<Array> backward(const std::vector<Array>& gys) override;
};

// Utility functions
Array reshape_sum_backward(const Array& gy, const std::pair<int, int>& x_shape, 
                          int axis, bool keepdims);
Array sum_to(const Array& x, const std::pair<int, int>& shape);

// User-facing functions
std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& x, float c);
std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& x, 
                                   int rows, int cols);
std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& x, 
                              int axis = -1, bool keepdims = false);
std::shared_ptr<Tensor> broadcast_to(const std::shared_ptr<Tensor>& x, 
                                       int rows, int cols);
std::shared_ptr<Tensor> sum_to_func(const std::shared_ptr<Tensor>& x, 
                                       int rows, int cols);
std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& x, 
                                 const std::shared_ptr<Tensor>& W);
std::shared_ptr<Tensor> mean_squared_error(const std::shared_ptr<Tensor>& x0, 
                                             const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> linear_func(const std::shared_ptr<Tensor>& x, 
                                       const std::shared_ptr<Tensor>& W, 
                                       const std::shared_ptr<Tensor>& b = nullptr);
std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& x);

// Operator overloads for Tensor
// Addition: x + y
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& x, float c);
std::shared_ptr<Tensor> operator+(float c, const std::shared_ptr<Tensor>& x);

// Subtraction: x - y
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x, float c);
std::shared_ptr<Tensor> operator-(float c, const std::shared_ptr<Tensor>& x);

// Unary minus: -x
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x);

// Multiplication: x * y
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& x, float c);
std::shared_ptr<Tensor> operator*(float c, const std::shared_ptr<Tensor>& x);

// Division: x / y
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& x, float c);
std::shared_ptr<Tensor> operator/(float c, const std::shared_ptr<Tensor>& x);

} // namespace miniTensor

#endif // MINITENSOR_FUNCTIONS_H

