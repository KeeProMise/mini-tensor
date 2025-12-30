#include <iostream>
#include "miniTensor/core.h"
#include "miniTensor/functions.h"
#include "miniTensor/layers.h"
#include "miniTensor/models.h"
#include "miniTensor/optimizers.h"

using namespace miniTensor;

int main() {
    std::cout << "MiniTensor C++ - Deep Learning Framework" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;

    // Example 1: Basic operations
    std::cout << "Example 1: Basic Tensor Operations" << std::endl;
    Array x_data(2, 2);
    x_data << 1.0f, 2.0f,
              3.0f, 4.0f;
    
    auto x = std::make_shared<Tensor>(x_data);
    std::cout << "x = " << std::endl << x->data << std::endl;
  
    auto y = add(x, x);
    std::cout << "x + x = " << std::endl << y->data << std::endl;
    
    auto z = mul(x, x);
    std::cout << "x * x = " << std::endl << z->data << std::endl << std::endl;

    // Example 2: Automatic differentiation
    std::cout << "Example 2: Automatic Differentiation" << std::endl;
    Array a_data(1, 1);
    a_data << 2.0f;
    auto a = std::make_shared<Tensor>(a_data);
    
    auto b = mul(a, a);  // b = a^2
    auto c = mul(b, a);  // c = a^3
    
    c->backward();
    std::cout << "a = " << a->data(0, 0) << std::endl;
    std::cout << "c = a^3 = " << c->data(0, 0) << std::endl;
    if (a->grad) {
        std::cout << "dc/da = " << a->grad->data(0, 0) << std::endl;
        std::cout << "Expected: 3*a^2 = " << 3.0f * 2.0f * 2.0f << std::endl << std::endl;
    } else {
        std::cout << "Error: a->grad is nullptr" << std::endl << std::endl;
    }

    // Example 3: Simple neural network
    std::cout << "Example 3: Simple Neural Network" << std::endl;
    auto model = std::make_shared<TwoLayerNet>(10, 1);
    
    Array input_data(1, 2);
    input_data << 1.0f, 2.0f;
    auto input = std::make_shared<Tensor>(input_data);
    
    auto output = model->forward(input);
    std::cout << "Input shape: (" << input->data.rows() << ", " << input->data.cols() << ")" << std::endl;
    std::cout << "Output shape: (" << output->data.rows() << ", " << output->data.cols() << ")" << std::endl;
    std::cout << "Output = " << std::endl << output->data << std::endl << std::endl;

    // Example 4: MLP Model
    std::cout << "Example 4: MLP Model" << std::endl;
    std::vector<int> sizes = {10, 5, 1};
    auto mlp = std::make_shared<MLP>(sizes);
    
    Array mlp_input(1, 2);
    mlp_input << 1.0f, 2.0f;
    auto mlp_x = std::make_shared<Tensor>(mlp_input);
    
    auto mlp_output = mlp->forward(mlp_x);
    std::cout << "MLP Input shape: (" << mlp_x->data.rows() << ", " << mlp_x->data.cols() << ")" << std::endl;
    std::cout << "MLP Output shape: (" << mlp_output->data.rows() << ", " << mlp_output->data.cols() << ")" << std::endl;
    std::cout << "MLP Output = " << std::endl << mlp_output->data << std::endl << std::endl;

    // Example 5: Optimizer
    std::cout << "Example 5: SGD Optimizer" << std::endl;
    auto optimizer = std::make_shared<SGD>(0.01f);
    optimizer->setup(mlp);
    
    // Create a dummy loss for demonstration
    Array target_data(1, 1);
    target_data << 0.5f;
    auto target = std::make_shared<Tensor>(target_data);
    
    auto loss = mean_squared_error(mlp_output, target);
    std::cout << "Loss = " << loss->data(0, 0) << std::endl;
    
    loss->backward();
    std::cout << "Backward pass completed" << std::endl;
    
    optimizer->update();
    std::cout << "Parameters updated" << std::endl << std::endl;

    std::cout << "All examples completed successfully!" << std::endl;
    
    return 0;
}

