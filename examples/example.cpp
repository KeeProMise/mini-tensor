#include "miniTensor/miniTensor.h"

#include <iostream>
#include <Eigen/Dense>

using namespace miniTensor;

int main() {
  // 创建张量
  Array data(2, 2);
  data << 1.0f, 2.0f, 3.0f, 4.0f;
  auto x = std::make_shared<Tensor>(data);

  auto y = 2 * x;
  auto z = y * x;

  // 自动微分
  z->backward(true);

  std::cout << "z = " << z->data << std::endl;
  std::cout << "y = " << y->data << std::endl;
  std::cout << "x = " << x->data << std::endl;
  std::cout << "dy = " << y->grad->data << std::endl;
  std::cout << "dx = " << x->grad->data << std::endl;
}