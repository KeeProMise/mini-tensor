#include "miniTensor/miniTensor.h"

#include <iostream>
#include <Eigen/Dense>

using namespace miniTensor;

int main() {
  // 创建张量
  Array data(2, 2);
  data << 1.0f, 2.0f, 3.0f, 4.0f;
  auto x = std::make_shared<Tensor>(data);
  float a = 2.0f;
  auto y = 2 * x;
  auto z = matmul(x, y);

  // 自动微分
  z->backward(true);

  std::cout << "z = " << z->data << std::endl;
  std::cout << "y = " << y << std::endl;
  std::cout << "x = " << x << std::endl;
  std::cout << "dy = " << y->grad->data << std::endl;
  std::cout << "dx = " << x->grad->data << std::endl;
  
  // 打印计算图
  std::cout << "\n";
  std::cout << "=== 计算图（文本格式）===\n";
  z->print_graph();
  
  std::cout << "\n";
  std::cout << "=== 计算图（DOT格式，可用于Graphviz）===\n";
  std::cout << z->to_dot() << std::endl;
  
  std::cout << "\n";
  std::cout << "=== 计算图（DOT格式，包含tensor值）===\n";
  std::cout << z->to_dot(true) << std::endl;
}