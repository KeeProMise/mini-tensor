# MiniTensor C++ Implementation

这是MiniTensor深度学习框架的C++实现版本。

## 依赖

- C++17 或更高版本
- CMake 3.10 或更高版本
- Eigen3 库

## 安装Eigen3

### macOS
```bash
brew install eigen
```

### Ubuntu/Debian
```bash
sudo apt-get install libeigen3-dev
```

### 从源码安装
```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build && cd build
cmake ..
sudo make install
```

## 编译

```bash
mkdir build
cd build
cmake ..
make
```

## 运行

```bash
./bin/miniTensor_example
```

## 项目结构

```
cpp/app/
├── CMakeLists.txt          # CMake构建配置
├── include/                 # 头文件目录
│   └── miniTensor/         # MiniTensor命名空间
│       ├── core.h          # 核心自动微分系统
│       ├── functions.h     # 数学函数
│       ├── layers.h        # 神经网络层
│       ├── models.h        # 模型定义
│       ├── optimizers.h    # 优化器
│       └── miniTensor.h    # 主头文件（包含所有功能）
├── src/                     # 源文件目录
│   ├── core.cpp
│   ├── functions.cpp
│   ├── layers.cpp
│   ├── models.cpp
│   └── optimizers.cpp
├── examples/                # 示例程序
│   └── main.cpp
└── README.md
```

## 使用库

在你的项目中使用MiniTensor库：

```cpp
#include "miniTensor/miniTensor.h"  // 或单独包含需要的头文件

using namespace miniTensor;

// 你的代码...
```

## 使用示例

```cpp
#include "miniTensor/miniTensor.h"  // 或单独包含需要的头文件
#include <Eigen/Dense>

using namespace miniTensor;

// 创建张量
Array data(2, 2);
data << 1.0f, 2.0f, 3.0f, 4.0f;
auto x = std::make_shared<Tensor>(data);

// 执行运算（使用操作符重载）
auto y = x + x;  // 或 add(x, x)
auto z = x * x;  // 或 mul(x, x)

// 自动微分
z->backward();
// x->grad 现在包含梯度
```

## 特性

- ✅ 自动微分（反向传播）
- ✅ 基础数学运算（加、减、乘、除、幂、指数等）
- ✅ 三角函数（sin, cos, tanh）
- ✅ 神经网络层（Linear）
- ✅ 多层感知机（MLP）
- ✅ 优化器（SGD, MomentumSGD）
- ✅ 损失函数（均方误差）

## 注意事项

- 当前实现使用Eigen库进行矩阵运算
- 某些高级特性（如广播、reshape等）已简化实现
- 内存管理使用智能指针（shared_ptr）

