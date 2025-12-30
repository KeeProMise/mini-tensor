# 为什么使用 shared_ptr？

## 问题背景

在自动微分框架中，`Variable` 需要保存创建它的 `Function` 的引用（通过 `creator` 指针），以便在反向传播时能够访问该函数的 `inputs` 和 `outputs`。

## 问题1：局部对象生命周期问题

### 错误的实现（使用原始指针）

```cpp
std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& x0, 
                              const std::shared_ptr<Variable>& x1) {
    Mul func;  // 局部对象
    auto result = func.call({x0, x1});
    // func.call() 内部会执行：
    //   result->creator = &func;  // 保存原始指针
    return result;
}  // ❌ func 在这里被销毁！result->creator 现在指向无效内存
```

**问题：**
- `func` 是局部对象，函数返回后立即被销毁
- `result->creator` 指向已销毁的对象（悬空指针）
- 当调用 `backward()` 时访问 `creator->inputs` 会导致未定义行为或崩溃

### 正确的实现（使用 shared_ptr）

```cpp
std::shared_ptr<Variable> mul(const std::shared_ptr<Variable>& x0, 
                              const std::shared_ptr<Variable>& x1) {
    auto func = std::make_shared<Mul>();  // 使用 shared_ptr
    auto result = func->call({x0, x1});
    // func->call() 内部会执行：
    //   result->creator = func;  // 保存 shared_ptr
    return result;
}  // ✅ func 的引用计数 > 0（被 result->creator 持有），不会被销毁
```

**优势：**
- `func` 的生命周期由引用计数管理
- `result->creator` 持有 `func` 的引用，确保 `func` 在需要时保持存活
- 当 `result` 和所有引用都被销毁时，`func` 才会被自动清理

## 问题2：多个 Variable 可能共享同一个 Function

在计算图中，一个 `Function` 可能被多个 `Variable` 引用：

```cpp
auto a = Variable(2);
auto b = mul(a, a);  // b->creator 指向 Mul 对象
auto c = mul(b, a);  // c->creator 指向另一个 Mul 对象
// 在 backward() 中，可能需要访问这些 Function 对象
```

使用 `shared_ptr` 可以安全地管理这种共享关系。

## 为什么不用其他方案？

### ❌ 原始指针 (Function*)
- 无法管理生命周期
- 容易产生悬空指针
- 需要手动管理内存（容易出错）

### ❌ unique_ptr
- 独占所有权，不能共享
- 多个 Variable 不能同时持有同一个 Function 的引用

### ✅ shared_ptr
- 共享所有权
- 自动管理生命周期
- 引用计数为 0 时自动销毁
- 完美适合这种"多个对象需要共享同一个资源"的场景

## 实现细节

### 1. Function 继承 enable_shared_from_this

```cpp
class Function : public std::enable_shared_from_this<Function> {
    // ...
};
```

这允许在 `Function::call()` 中获取指向自身的 `shared_ptr`：

```cpp
std::shared_ptr<Function> self_ptr = shared_from_this();
output->set_creator(self_ptr);  // Variable 持有 Function 的引用
```

### 2. Variable 使用 shared_ptr 存储 creator

```cpp
class Variable {
    std::shared_ptr<Function> creator;  // 而不是 Function*
    // ...
};
```

这样确保了 `Function` 对象在 `Variable` 存在期间保持存活。

## 生命周期示例

```cpp
auto a = Variable(2);
auto b = mul(a, a);  
// b->creator 持有 Mul 的 shared_ptr，引用计数 = 1

auto c = mul(b, a);
// c->creator 持有另一个 Mul 的 shared_ptr，引用计数 = 1

c->backward();
// backward() 中会访问 c->creator->inputs 和 c->creator->outputs
// 因为使用了 shared_ptr，Function 对象仍然有效

// 当 c 被销毁时，c->creator 的引用计数变为 0
// Mul 对象被自动销毁
```

## 总结

使用 `shared_ptr` 是为了：
1. **解决生命周期问题**：确保 `Function` 对象在需要时保持存活
2. **支持共享所有权**：多个 `Variable` 可以安全地引用同一个 `Function`
3. **自动内存管理**：避免手动管理内存，减少内存泄漏和悬空指针的风险
4. **符合 C++ 最佳实践**：使用智能指针管理资源

这是 C++ 中处理"对象需要被多个其他对象共享"场景的标准做法。

