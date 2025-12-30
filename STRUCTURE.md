# DeZero C++ 项目结构说明

本项目按照标准的C++项目结构组织，符合CMake最佳实践。

## 目录结构

```
cpp/app/
├── CMakeLists.txt          # CMake构建配置文件
├── README.md               # 项目说明文档
├── STRUCTURE.md           # 本文件 - 项目结构说明
├── .gitignore             # Git忽略文件
├── .clang-format          # 代码格式化配置
│
├── include/                # 公共头文件目录
│   └── dezero/            # 命名空间目录
│       ├── core.h         # 核心自动微分系统
│       ├── functions.h    # 数学函数定义
│       ├── layers.h       # 神经网络层定义
│       ├── models.h       # 模型定义
│       ├── optimizers.h   # 优化器定义
│       └── dezero.h       # 主头文件（包含所有功能）
│
├── src/                    # 源文件目录
│   ├── core.cpp           # 核心实现
│   ├── functions.cpp      # 数学函数实现
│   ├── layers.cpp         # 神经网络层实现
│   ├── models.cpp         # 模型实现
│   └── optimizers.cpp     # 优化器实现
│
└── examples/               # 示例程序目录
    └── main.cpp           # 示例程序
```

## 设计原则

### 1. 头文件组织
- 所有公共头文件放在 `include/dezero/` 目录下
- 头文件使用命名空间 `dezero`
- 头文件之间使用相对路径引用（如 `#include "core.h"`）
- 提供 `dezero.h` 作为统一入口，包含所有功能

### 2. 源文件组织
- 所有实现文件放在 `src/` 目录下
- 源文件使用完整路径引用头文件（如 `#include "dezero/core.h"`）
- 源文件与头文件一一对应

### 3. CMake配置
- 使用现代CMake语法（CMake 3.10+）
- 正确设置include目录
- 分离库和可执行文件
- 支持安装（install）功能
- 使用 `target_include_directories` 管理依赖

### 4. 命名规范
- 头文件使用 `.h` 扩展名
- 源文件使用 `.cpp` 扩展名
- 头文件保护宏使用 `DEZERO_*_H` 格式
- 类名使用PascalCase
- 函数名使用snake_case

## 编译和使用

### 编译库
```bash
mkdir build && cd build
cmake ..
make
```

### 使用库
```cpp
// 方式1: 包含主头文件（推荐）
#include "dezero/dezero.h"

// 方式2: 按需包含
#include "dezero/core.h"
#include "dezero/functions.h"
```

### 链接库
在CMakeLists.txt中：
```cmake
target_link_libraries(your_target dezero)
target_include_directories(your_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

## 扩展项目

添加新功能时：
1. 在 `include/dezero/` 添加头文件
2. 在 `src/` 添加对应的实现文件
3. 在 `CMakeLists.txt` 的 `DEZERO_SOURCES` 和 `DEZERO_HEADERS` 中添加新文件
4. 如需导出，在 `dezero.h` 中包含新头文件

## 符合的标准

- ✅ C++17标准
- ✅ CMake 3.10+ 最佳实践
- ✅ 标准C++项目目录结构
- ✅ 现代CMake语法
- ✅ 清晰的依赖管理

