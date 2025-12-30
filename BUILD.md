# 构建说明

## 前置要求

### 1. 安装 Eigen3

#### macOS
```bash
brew install eigen
```

如果遇到权限问题，可能需要：
```bash
sudo chown -R $(whoami) /opt/homebrew/Cellar
brew install eigen
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libeigen3-dev
```

#### 手动安装
如果无法使用包管理器，可以从源码安装：

```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build && cd build
cmake ..
sudo make install
```

### 2. 验证 Eigen3 安装

安装后，Eigen3 通常位于以下位置之一：
- `/usr/local/include/eigen3`
- `/opt/homebrew/include/eigen3` (Apple Silicon Mac)
- `/usr/include/eigen3`

验证方法：
```bash
ls /usr/local/include/eigen3/Eigen/Dense
# 或
ls /opt/homebrew/include/eigen3/Eigen/Dense
```

## 构建项目

### 标准构建

```bash
cd cpp/app
mkdir build && cd build
cmake ..
make
```

### 如果 Eigen3 不在标准路径

如果 CMake 无法自动找到 Eigen3，可以手动指定路径：

```bash
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
make
```

例如：
```bash
# macOS (Intel)
cmake -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 ..

# macOS (Apple Silicon)
cmake -DEIGEN3_INCLUDE_DIR=/opt/homebrew/include/eigen3 ..

# Linux
cmake -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 ..
```

### 使用 Ninja 构建（更快）

```bash
cd build
cmake -GNinja ..
ninja
```

### 调试构建

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### 发布构建

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## 运行示例

构建成功后：

```bash
./bin/dezero_example
```

## 常见问题

### 问题1: Eigen3 not found

**解决方案：**
1. 确保已安装 Eigen3（见上方安装说明）
2. 如果已安装但 CMake 找不到，手动指定路径：
   ```bash
   cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
   ```

### 问题2: 编译错误 "Eigen/Dense file not found"

**解决方案：**
- 检查 Eigen3 是否正确安装
- 确认 CMake 配置时显示了正确的 Eigen3 路径
- 尝试手动指定路径

### 问题3: 链接错误

**解决方案：**
- Eigen3 是 header-only 库，不需要链接
- 如果仍有问题，检查 CMakeLists.txt 中的 include 路径设置

## 验证构建

构建成功后，应该生成以下文件：
- `build/libdezero.a` - 静态库
- `build/bin/dezero_example` - 示例可执行文件

