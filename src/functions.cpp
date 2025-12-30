#include "miniTensor/functions.h"
#include "miniTensor/core.h"
#include <cmath>
#include <algorithm>

namespace miniTensor {

// Add implementation
std::vector<Array> Add::forward(const std::vector<Array>& xs) {
    x0_shape = {xs[0].rows(), xs[0].cols()};
    x1_shape = {xs[1].rows(), xs[1].cols()};
    
    // 处理标量广播
    Array x0 = xs[0];
    Array x1 = xs[1];
    
    if (x0.rows() == 1 && x0.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x0 = Array::Constant(x1.rows(), x1.cols(), x0(0, 0));
        x0_shape = {x1.rows(), x1.cols()};
    }
    else if (x1.rows() == 1 && x1.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x1 = Array::Constant(x0.rows(), x0.cols(), x1(0, 0));
        x1_shape = {x0.rows(), x0.cols()};
    }
    
    Array y = x0 + x1;
    return {y};
}

std::vector<Array> Add::backward(const std::vector<Array>& gys) {
    if (gys.empty()) {
        return {Array::Zero(inputs[0]->data.rows(), inputs[0]->data.cols()),
                Array::Zero(inputs[1]->data.rows(), inputs[1]->data.cols())};
    }
    Array gy = gys[0];
    Array gx0 = gy;
    Array gx1 = gy;
    
    // 如果原始输入是标量，需要sum到1x1
    if (x0_shape.first == 1 && x0_shape.second == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        gx0 = sum_to(gy, x0_shape);
    }
    if (x1_shape.first == 1 && x1_shape.second == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        gx1 = sum_to(gy, x1_shape);
    }
    
    return {gx0, gx1};
}

// Mul implementation
std::vector<Array> Mul::forward(const std::vector<Array>& xs) {
    x0_shape = {xs[0].rows(), xs[0].cols()};
    x1_shape = {xs[1].rows(), xs[1].cols()};
    
    // 处理标量广播：如果一个是1x1，广播到另一个的形状
    Array x0 = xs[0];
    Array x1 = xs[1];
    
    // 如果x0是标量(1x1)，广播到x1的形状
    if (x0.rows() == 1 && x0.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x0 = Array::Constant(x1.rows(), x1.cols(), x0(0, 0));
        x0_shape = {x1.rows(), x1.cols()};
    }
    // 如果x1是标量(1x1)，广播到x0的形状
    else if (x1.rows() == 1 && x1.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x1 = Array::Constant(x0.rows(), x0.cols(), x1(0, 0));
        x1_shape = {x0.rows(), x0.cols()};
    }
    
    Array y = x0.cwiseProduct(x1);
    return {y};
}

std::vector<Array> Mul::backward(const std::vector<Array>& gys) {
    if (gys.empty() || gys[0].size() == 0) {
        return {Array::Zero(inputs[0]->data.rows(), inputs[0]->data.cols()),
                Array::Zero(inputs[1]->data.rows(), inputs[1]->data.cols())};
    }
    Array gy = gys[0];
    Array x0 = inputs[0]->data;
    Array x1 = inputs[1]->data;
    
    // 处理标量广播：如果输入是标量，需要广播到gy的形状
    if (x0.rows() == 1 && x0.cols() == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        x0 = Array::Constant(gy.rows(), gy.cols(), x0(0, 0));
    }
    if (x1.rows() == 1 && x1.cols() == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        x1 = Array::Constant(gy.rows(), gy.cols(), x1(0, 0));
    }
    
    Array gx0 = gy.cwiseProduct(x1);
    Array gx1 = gy.cwiseProduct(x0);
    
    // 如果原始输入是标量，需要sum到1x1
    if (x0_shape.first == 1 && x0_shape.second == 1 && (gx0.rows() != 1 || gx0.cols() != 1)) {
        gx0 = sum_to(gx0, x0_shape);
    }
    if (x1_shape.first == 1 && x1_shape.second == 1 && (gx1.rows() != 1 || gx1.cols() != 1)) {
        gx1 = sum_to(gx1, x1_shape);
    }
    
    return {gx0, gx1};
}

// Neg implementation
std::vector<Array> Neg::forward(const std::vector<Array>& xs) {
    return {-xs[0]};
}

std::vector<Array> Neg::backward(const std::vector<Array>& gys) {
    return {-gys[0]};
}

// Sub implementation
std::vector<Array> Sub::forward(const std::vector<Array>& xs) {
    x0_shape = {xs[0].rows(), xs[0].cols()};
    x1_shape = {xs[1].rows(), xs[1].cols()};
    
    // 处理标量广播
    Array x0 = xs[0];
    Array x1 = xs[1];
    
    if (x0.rows() == 1 && x0.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x0 = Array::Constant(x1.rows(), x1.cols(), x0(0, 0));
        x0_shape = {x1.rows(), x1.cols()};
    }
    else if (x1.rows() == 1 && x1.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x1 = Array::Constant(x0.rows(), x0.cols(), x1(0, 0));
        x1_shape = {x0.rows(), x0.cols()};
    }
    
    Array y = x0 - x1;
    return {y};
}

std::vector<Array> Sub::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array gx0 = gy;
    Array gx1 = -gy;
    
    // 如果原始输入是标量，需要sum到1x1
    if (x0_shape.first == 1 && x0_shape.second == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        gx0 = sum_to(gy, x0_shape);
    }
    if (x1_shape.first == 1 && x1_shape.second == 1 && (gy.rows() != 1 || gy.cols() != 1)) {
        gx1 = sum_to(-gy, x1_shape);
    }
    
    return {gx0, gx1};
}

// Div implementation
std::vector<Array> Div::forward(const std::vector<Array>& xs) {
    // 处理标量广播
    Array x0 = xs[0];
    Array x1 = xs[1];
    
    if (x0.rows() == 1 && x0.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x0 = Array::Constant(x1.rows(), x1.cols(), x0(0, 0));
    }
    else if (x1.rows() == 1 && x1.cols() == 1 && (x0.rows() != x1.rows() || x0.cols() != x1.cols())) {
        x1 = Array::Constant(x0.rows(), x0.cols(), x1(0, 0));
    }
    
    Array y = x0.cwiseQuotient(x1);
    return {y};
}

std::vector<Array> Div::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x0 = inputs[0]->data;
    Array x1 = inputs[1]->data;
    
    Array gx0 = gy.cwiseQuotient(x1);
    Array gx1 = -gy.cwiseProduct(x0).cwiseQuotient(x1.cwiseProduct(x1));
    
    auto x0_shape = inputs[0]->shape();
    auto x1_shape = inputs[1]->shape();
    if (x0_shape != x1_shape) {
        gx0 = sum_to(gx0, x0_shape);
        gx1 = sum_to(gx1, x1_shape);
    }
    
    return {gx0, gx1};
}

// Pow implementation
std::vector<Array> Pow::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().pow(c).matrix();
    return {y};
}

std::vector<Array> Pow::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x = inputs[0]->data;
    Array gx = c * x.array().pow(c - 1).matrix().cwiseProduct(gy);
    return {gx};
}

// Exp implementation
std::vector<Array> Exp::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().exp().matrix();
    return {y};
}

std::vector<Array> Exp::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    if (!outputs.empty() && !outputs[0].expired()) {
        auto output = outputs[0].lock();
        Array y = output->data;
        Array gx = gy.cwiseProduct(y);
        return {gx};
    }
    Array x = inputs[0]->data;
    Array y = x.array().exp().matrix();
    Array gx = gy.cwiseProduct(y);
    return {gx};
}

// Sin implementation
std::vector<Array> Sin::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().sin().matrix();
    return {y};
}

std::vector<Array> Sin::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    auto x_var = inputs[0];
    auto cos_x = std::make_shared<Tensor>(x_var->data);
    auto cos_result = cos(cos_x);
    Array gx = gy.cwiseProduct(cos_result->data);
    return {gx};
}

// Cos implementation
std::vector<Array> Cos::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().cos().matrix();
    return {y};
}

std::vector<Array> Cos::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    auto x_var = inputs[0];
    auto sin_x = std::make_shared<Tensor>(x_var->data);
    auto sin_result = sin(sin_x);
    Array gx = -gy.cwiseProduct(sin_result->data);
    return {gx};
}

// Tanh implementation
std::vector<Array> Tanh::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().tanh().matrix();
    return {y};
}

std::vector<Array> Tanh::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    if (!outputs.empty() && !outputs[0].expired()) {
        auto output = outputs[0].lock();
        Array y = output->data;
        Array ones = Array::Ones(y.rows(), y.cols());
        Array gx = gy.cwiseProduct(ones - y.cwiseProduct(y));
        return {gx};
    }
    return {gy};
}

// Reshape implementation
std::vector<Array> Reshape::forward(const std::vector<Array>& xs) {
    x_shape = {xs[0].rows(), xs[0].cols()};
    Array y = xs[0].reshaped(shape.first, shape.second);
    return {y};
}

std::vector<Array> Reshape::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array gx = gy.reshaped(x_shape.first, x_shape.second);
    return {gx};
}

// Transpose implementation
std::vector<Array> Transpose::forward(const std::vector<Array>& xs) {
    Array y = xs[0].transpose();
    return {y};
}

std::vector<Array> Transpose::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array gx = gy.transpose();
    return {gx};
}

// Sum implementation
std::vector<Array> Sum::forward(const std::vector<Array>& xs) {
    x_shape = {xs[0].rows(), xs[0].cols()};
    Array y;
    if (axis == -1) {
        y = Array::Constant(1, 1, xs[0].sum());
    } else if (axis == 0) {
        y = xs[0].colwise().sum();
    } else if (axis == 1) {
        y = xs[0].rowwise().sum();
    } else {
        y = Array::Constant(1, 1, xs[0].sum());
    }
    return {y};
}

std::vector<Array> Sum::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    gy = reshape_sum_backward(gy, x_shape, axis, keepdims);
    // Broadcast gy to x_shape
    if (gy.rows() == 1 && gy.cols() == 1) {
        Array gx = Array::Constant(x_shape.first, x_shape.second, gy(0, 0));
        return {gx};
    }
    return {gy};
}

// BroadcastTo implementation
std::vector<Array> BroadcastTo::forward(const std::vector<Array>& xs) {
    x_shape = {xs[0].rows(), xs[0].cols()};
    // Simple broadcasting - replicate if needed
    int row_rep = (shape.first + x_shape.first - 1) / x_shape.first;
    int col_rep = (shape.second + x_shape.second - 1) / x_shape.second;
    if (row_rep < 1) row_rep = 1;
    if (col_rep < 1) col_rep = 1;
    Array y = xs[0].replicate(row_rep, col_rep);
    // Crop to exact shape if needed
    if (y.rows() > shape.first || y.cols() > shape.second) {
        y = y.block(0, 0, shape.first, shape.second);
    }
    return {y};
}

std::vector<Array> BroadcastTo::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array gx = sum_to(gy, x_shape);
    return {gx};
}

// SumTo implementation
std::vector<Array> SumTo::forward(const std::vector<Array>& xs) {
    x_shape = {xs[0].rows(), xs[0].cols()};
    Array y = sum_to(xs[0], shape);
    return {y};
}

std::vector<Array> SumTo::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    // Broadcast gy back to x_shape
    if (gy.rows() == 1 && gy.cols() == 1) {
        Array gx = Array::Constant(x_shape.first, x_shape.second, gy(0, 0));
        return {gx};
    }
    return {gy};
}

// MatMul implementation
std::vector<Array> MatMul::forward(const std::vector<Array>& xs) {
    Array y = xs[0] * xs[1];
    return {y};
}

std::vector<Array> MatMul::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x = inputs[0]->data;
    Array W = inputs[1]->data;
    
    Array gx = gy * W.transpose();
    Array gW = x.transpose() * gy;
    
    return {gx, gW};
}

// MeanSquaredError implementation
std::vector<Array> MeanSquaredError::forward(const std::vector<Array>& xs) {
    Array diff = xs[0] - xs[1];
    float y_val = diff.cwiseProduct(diff).sum() / diff.size();
    Array y = Array::Constant(1, 1, y_val);
    return {y};
}

std::vector<Array> MeanSquaredError::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x0 = inputs[0]->data;
    Array x1 = inputs[1]->data;
    Array diff = x0 - x1;
    int n = diff.size();
    Array gx0 = gy(0, 0) * 2.0f * diff / n;
    Array gx1 = -gx0;
    return {gx0, gx1};
}

// Log implementation
std::vector<Array> Log::forward(const std::vector<Array>& xs) {
    Array y = xs[0].array().log().matrix();
    return {y};
}

std::vector<Array> Log::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x = inputs[0]->data;
    Array gx = gy.cwiseQuotient(x);
    return {gx};
}

// LinearFunction implementation
std::vector<Array> LinearFunction::forward(const std::vector<Array>& xs) {
    Array y = xs[0] * xs[1];
    if (xs.size() > 2 && xs[2].size() > 0) {
        // Add bias: y = x*W + b
        // b is (1, out_size), need to broadcast to (batch, out_size)
        for (int i = 0; i < y.rows(); ++i) {
            y.row(i) += xs[2].row(0);
        }
    }
    return {y};
}

std::vector<Array> LinearFunction::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    Array x = inputs[0]->data;
    Array W = inputs[1]->data;
    
    Array gx = gy * W.transpose();
    Array gW = x.transpose() * gy;
    
    if (inputs.size() > 2 && inputs[2]) {
        Array gb = gy.colwise().sum();
        if (gb.rows() != inputs[2]->data.rows() || gb.cols() != inputs[2]->data.cols()) {
            gb = gb.row(0);  // Take first row if needed
        }
        return {gx, gW, gb};
    }
    
    return {gx, gW};
}

// Sigmoid implementation
std::vector<Array> Sigmoid::forward(const std::vector<Array>& xs) {
    Array exp_neg_x = (-xs[0]).array().exp().matrix();
    Array y = Array::Ones(xs[0].rows(), xs[0].cols()).cwiseQuotient(
        Array::Ones(xs[0].rows(), xs[0].cols()) + exp_neg_x);
    return {y};
}

std::vector<Array> Sigmoid::backward(const std::vector<Array>& gys) {
    Array gy = gys[0];
    if (!outputs.empty() && !outputs[0].expired()) {
        auto output = outputs[0].lock();
        Array y = output->data;
        Array ones = Array::Ones(y.rows(), y.cols());
        Array gx = gy.cwiseProduct(y).cwiseProduct(ones - y);
        return {gx};
    }
    return {gy};
}

// Utility functions
Array reshape_sum_backward(const Array& gy, const std::pair<int, int>& x_shape, 
                          int axis, bool keepdims) {
    // Simplified implementation
    return gy;
}

Array sum_to(const Array& x, const std::pair<int, int>& shape) {
    // Simplified sum_to - just reshape if same size, otherwise sum
    if (x.rows() == shape.first && x.cols() == shape.second) {
        return x;
    }
    // Sum to target shape
    Array y = Array::Zero(shape.first, shape.second);
    float sum_val = x.sum();
    if (shape.first == 1 && shape.second == 1) {
        y(0, 0) = sum_val;
    }
    return y;
}

// User-facing functions
std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1) {
    auto func = std::make_shared<Add>();
    return func->call({x0, x1});
}

std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1) {
    auto func = std::make_shared<Mul>();
    return func->call({x0, x1});
}

std::shared_ptr<Tensor> neg(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Neg>();
    return func->call({x});
}

std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1) {
    auto func = std::make_shared<Sub>();
    return func->call({x0, x1});
}

std::shared_ptr<Tensor> div(const std::shared_ptr<Tensor>& x0, 
                              const std::shared_ptr<Tensor>& x1) {
    auto func = std::make_shared<Div>();
    return func->call({x0, x1});
}

std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& x, float c) {
    auto func = std::make_shared<Pow>(c);
    return func->call({x});
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Exp>();
    return func->call({x});
}

std::shared_ptr<Tensor> sin(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Sin>();
    return func->call({x});
}

std::shared_ptr<Tensor> cos(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Cos>();
    return func->call({x});
}

std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Tanh>();
    return func->call({x});
}

std::shared_ptr<Tensor> reshape(const std::shared_ptr<Tensor>& x, 
                                   int rows, int cols) {
    if (x->shape() == std::make_pair(rows, cols)) {
        return x;
    }
    auto func = std::make_shared<Reshape>(rows, cols);
    return func->call({x});
}

std::shared_ptr<Tensor> transpose(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Transpose>();
    return func->call({x});
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& x, 
                              int axis, bool keepdims) {
    auto func = std::make_shared<Sum>(axis, keepdims);
    return func->call({x});
}

std::shared_ptr<Tensor> broadcast_to(const std::shared_ptr<Tensor>& x, 
                                       int rows, int cols) {
    if (x->shape() == std::make_pair(rows, cols)) {
        return x;
    }
    auto func = std::make_shared<BroadcastTo>(rows, cols);
    return func->call({x});
}

std::shared_ptr<Tensor> sum_to_func(const std::shared_ptr<Tensor>& x, 
                                       int rows, int cols) {
    if (x->shape() == std::make_pair(rows, cols)) {
        return x;
    }
    auto func = std::make_shared<SumTo>(rows, cols);
    return func->call({x});
}

std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& x, 
                                 const std::shared_ptr<Tensor>& W) {
    auto func = std::make_shared<MatMul>();
    return func->call({x, W});
}

std::shared_ptr<Tensor> mean_squared_error(const std::shared_ptr<Tensor>& x0, 
                                             const std::shared_ptr<Tensor>& x1) {
    auto func = std::make_shared<MeanSquaredError>();
    return func->call({x0, x1});
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Log>();
    return func->call({x});
}

std::shared_ptr<Tensor> linear_func(const std::shared_ptr<Tensor>& x, 
                                      const std::shared_ptr<Tensor>& W, 
                                      const std::shared_ptr<Tensor>& b) {
    auto func = std::make_shared<LinearFunction>();
    if (b) {
        return func->call({x, W, b});
    } else {
        return func->call({x, W});
    }
}

std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& x) {
    auto func = std::make_shared<Sigmoid>();
    return func->call({x});
}

// Operator overloads implementation
// Addition: x + y
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1) {
    return add(x0, x1);
}

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& x, float c) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return add(x, c_tensor);
}

std::shared_ptr<Tensor> operator+(float c, const std::shared_ptr<Tensor>& x) {
    return operator+(x, c);  // Commutative
}

// Subtraction: x - y
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1) {
    return sub(x0, x1);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x, float c) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return sub(x, c_tensor);
}

std::shared_ptr<Tensor> operator-(float c, const std::shared_ptr<Tensor>& x) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return sub(c_tensor, x);
}

// Unary minus: -x
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& x) {
    return neg(x);
}

// Multiplication: x * y
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1) {
    return mul(x0, x1);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& x, float c) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return mul(x, c_tensor);
}

std::shared_ptr<Tensor> operator*(float c, const std::shared_ptr<Tensor>& x) {
    return operator*(x, c);  // Commutative
}

// Division: x / y
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& x0, 
                                  const std::shared_ptr<Tensor>& x1) {
    return div(x0, x1);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& x, float c) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return div(x, c_tensor);
}

std::shared_ptr<Tensor> operator/(float c, const std::shared_ptr<Tensor>& x) {
    Array c_array = as_array(c);
    auto c_tensor = std::make_shared<Tensor>(c_array);
    return div(c_tensor, x);
}

} // namespace miniTensor

