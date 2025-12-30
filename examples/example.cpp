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
  auto y = a * x;
  auto z = matmul(y, x);

  // 自动微分
  z->backward(true);

  std::cout << "z = " << z << std::endl;
  std::cout << "y = " << y << std::endl;
  std::cout << "x = " << x << std::endl;
}