#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include "miniTensor/miniTensor.h"

using namespace miniTensor;

// 生成XOR问题的训练数据
void generate_xor_data(int n_samples, 
                       std::vector<Array>& X_train, 
                       std::vector<Array>& y_train) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    
    X_train.clear();
    y_train.clear();
    
    for (int i = 0; i < n_samples; ++i) {
        int x1 = dis(gen);
        int x2 = dis(gen);
        int label = x1 ^ x2;  // XOR操作
        
        Array x(1, 2);
        x << static_cast<float>(x1), static_cast<float>(x2);
        X_train.push_back(x);
        
        Array y(1, 1);
        y << static_cast<float>(label);
        y_train.push_back(y);
    }
}

// 生成简单的回归数据 (y = sin(x) + noise)
void generate_regression_data(int n_samples,
                              std::vector<Array>& X_train,
                              std::vector<Array>& y_train) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_dis(-3.14f, 3.14f);
    std::normal_distribution<float> noise_dis(0.0f, 0.1f);
    
    X_train.clear();
    y_train.clear();
    
    for (int i = 0; i < n_samples; ++i) {
        float x_val = x_dis(gen);
        float y_val = std::sin(x_val) + noise_dis(gen);
        
        Array x(1, 1);
        x << x_val;
        X_train.push_back(x);
        
        Array y(1, 1);
        y << y_val;
        y_train.push_back(y);
    }
}

// 计算准确率（用于分类任务）
float calculate_accuracy(const std::vector<Array>& predictions, 
                        const std::vector<Array>& targets) {
    int correct = 0;
    int total = predictions.size();
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        float pred = predictions[i](0, 0) > 0.5f ? 1.0f : 0.0f;
        float target = targets[i](0, 0);
        if (std::abs(pred - target) < 0.1f) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / static_cast<float>(total);
}

// 计算平均损失
float calculate_mean_loss(const std::vector<float>& losses) {
    float sum = 0.0f;
    for (float loss : losses) {
        sum += loss;
    }
    return sum / losses.size();
}

// XOR分类任务训练示例
void train_xor_classifier() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "DNN训练示例: XOR分类任务" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 生成训练数据
    std::vector<Array> X_train, y_train;
    std::vector<Array> X_test, y_test;
    
    generate_xor_data(1000, X_train, y_train);
    generate_xor_data(100, X_test, y_test);
    
    std::cout << "训练样本数: " << X_train.size() << std::endl;
    std::cout << "测试样本数: " << X_test.size() << std::endl;
    
    // 创建MLP模型: 输入2维 -> 隐藏层10维 -> 隐藏层5维 -> 输出1维
    std::vector<int> layer_sizes = {10, 5, 1};
    auto model = std::make_shared<MLP>(layer_sizes, sigmoid);
    
    // 创建优化器
    float learning_rate = 0.01f;  // 降低学习率避免梯度爆炸
    auto optimizer = std::make_shared<SGD>(learning_rate);
    optimizer->setup(model);
    
    // 训练参数
    int epochs = 100;
    int batch_size = 8;
    int n_batches = (X_train.size() + batch_size - 1) / batch_size;
    
    std::cout << "训练参数:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Batch Size: " << batch_size << std::endl;
    std::cout << "  - Learning Rate: " << learning_rate << std::endl;
    std::cout << "  - 模型结构: 2 -> 10 -> 5 -> 1" << std::endl;
    std::cout << "\n开始训练...\n" << std::endl;
    
    // 训练循环
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> epoch_losses;
        
        // 每个batch
        for (int batch = 0; batch < n_batches; ++batch) {
            // 准备batch数据
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(X_train.size()));
            
            // 累积梯度（简单的小批量处理）
            float batch_loss = 0.0f;
            int samples_in_batch = 0;
            
            // 在batch开始时清除梯度
            model->cleargrads();
            
            for (int i = start_idx; i < end_idx; ++i) {
                // 前向传播
                auto x = std::make_shared<Tensor>(X_train[i]);
                auto y_true = std::make_shared<Tensor>(y_train[i]);
                
                auto y_pred = model->forward(x);
                auto loss = mean_squared_error(y_pred, y_true);
                
                batch_loss += loss->data(0, 0);
                samples_in_batch++;
                
                // 反向传播（梯度会累积到参数上）
                loss->backward();
            }
            
            // 更新参数（使用累积的梯度）
            if (samples_in_batch > 0) {
                optimizer->update();
                epoch_losses.push_back(batch_loss / samples_in_batch);
            }
        }
        
        // 每10个epoch打印一次训练信息
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            float avg_loss = calculate_mean_loss(epoch_losses);
            
            // 在测试集上评估
            std::vector<Array> test_predictions;
            float test_loss = 0.0f;
            
            for (size_t i = 0; i < X_test.size(); ++i) {
                auto x = std::make_shared<Tensor>(X_test[i]);
                auto y_true = std::make_shared<Tensor>(y_test[i]);
                
                auto y_pred = model->forward(x);
                test_predictions.push_back(y_pred->data);
                test_loss += mean_squared_error(y_pred, y_true)->data(0, 0);
            }
            
            test_loss /= X_test.size();
            float accuracy = calculate_accuracy(test_predictions, y_test);
            
            std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << epochs << "] "
                      << "Train Loss: " << std::fixed << std::setprecision(4) << avg_loss
                      << "  Test Loss: " << std::setprecision(4) << test_loss
                      << "  Test Acc: " << std::setprecision(2) << accuracy * 100.0f << "%"
                      << std::endl;
        }
    }
    
    // 最终测试
    std::cout << "\n最终测试结果:" << std::endl;
    std::vector<Array> final_predictions;
    for (size_t i = 0; i < X_test.size(); ++i) {
        auto x = std::make_shared<Tensor>(X_test[i]);
        auto y_pred = model->forward(x);
        final_predictions.push_back(y_pred->data);
    }
    
    float final_accuracy = calculate_accuracy(final_predictions, y_test);
    std::cout << "测试准确率: " << std::fixed << std::setprecision(2) 
              << final_accuracy * 100.0f << "%" << std::endl;
    
    // 显示一些预测样例
    std::cout << "\n预测样例:" << std::endl;
    for (int i = 0; i < std::min(4, static_cast<int>(X_test.size())); ++i) {
        auto x = std::make_shared<Tensor>(X_test[i]);
        auto y_pred = model->forward(x);
        float pred_val = y_pred->data(0, 0);
        float true_val = y_test[i](0, 0);
        
        std::cout << "  输入: [" << X_test[i](0, 0) << ", " << X_test[i](0, 1) << "] "
                  << "真实值: " << true_val << " "
                  << "预测值: " << std::setprecision(3) << pred_val
                  << " (预测类别: " << (pred_val > 0.5f ? 1 : 0) << ")" << std::endl;
    }
}

// 回归任务训练示例
void train_regression_model() {
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "DNN训练示例: 回归任务 (y = sin(x))" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 生成训练数据
    std::vector<Array> X_train, y_train;
    std::vector<Array> X_test, y_test;
    
    generate_regression_data(500, X_train, y_train);
    generate_regression_data(100, X_test, y_test);
    
    std::cout << "训练样本数: " << X_train.size() << std::endl;
    std::cout << "测试样本数: " << X_test.size() << std::endl;
    
    // 创建MLP模型: 输入1维 -> 隐藏层20维 -> 隐藏层10维 -> 输出1维
    std::vector<int> layer_sizes = {20, 10, 1};
    auto model = std::make_shared<MLP>(layer_sizes, 
        [](const std::shared_ptr<Tensor>& x) { return tanh(x); });
    
    // 创建优化器（使用MomentumSGD）
    float learning_rate = 0.001f;  // 降低学习率避免梯度爆炸
    float momentum = 0.9f;
    auto optimizer = std::make_shared<MomentumSGD>(learning_rate, momentum);
    optimizer->setup(model);
    
    // 训练参数
    int epochs = 2000;
    int batch_size = 10;
    int n_batches = (X_train.size() + batch_size - 1) / batch_size;
    
    std::cout << "训练参数:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Batch Size: " << batch_size << std::endl;
    std::cout << "  - Learning Rate: " << learning_rate << std::endl;
    std::cout << "  - Momentum: " << momentum << std::endl;
    std::cout << "  - 模型结构: 1 -> 20 -> 10 -> 1" << std::endl;
    std::cout << "\n开始训练...\n" << std::endl;
    
    // 训练循环
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> epoch_losses;
        
        // 每个batch
        for (int batch = 0; batch < n_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, static_cast<int>(X_train.size()));
            
            float batch_loss = 0.0f;
            int samples_in_batch = 0;
            
            // 在batch开始时清除梯度
            model->cleargrads();
            
            for (int i = start_idx; i < end_idx; ++i) {
                // 前向传播
                auto x = std::make_shared<Tensor>(X_train[i]);
                auto y_true = std::make_shared<Tensor>(y_train[i]);
                
                auto y_pred = model->forward(x);
                auto loss = mean_squared_error(y_pred, y_true);
                
                batch_loss += loss->data(0, 0);
                samples_in_batch++;
                
                // 反向传播（梯度会累积到参数上）
                loss->backward();
            }
            
            // 更新参数（使用累积的梯度）
            if (samples_in_batch > 0) {
                optimizer->update();
                epoch_losses.push_back(batch_loss / samples_in_batch);
            }
        }
        
        // 每20个epoch打印一次训练信息
        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            float avg_loss = calculate_mean_loss(epoch_losses);
            
            // 在测试集上评估
            float test_loss = 0.0f;
            for (size_t i = 0; i < X_test.size(); ++i) {
                auto x = std::make_shared<Tensor>(X_test[i]);
                auto y_true = std::make_shared<Tensor>(y_test[i]);
                
                auto y_pred = model->forward(x);
                test_loss += mean_squared_error(y_pred, y_true)->data(0, 0);
            }
            
            test_loss /= X_test.size();
            
            std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << epochs << "] "
                      << "Train Loss: " << std::fixed << std::setprecision(6) << avg_loss
                      << "  Test Loss: " << std::setprecision(6) << test_loss
                      << std::endl;
        }
    }
    
    // 最终测试和可视化
    std::cout << "\n最终测试结果:" << std::endl;
    float final_test_loss = 0.0f;
    for (size_t i = 0; i < X_test.size(); ++i) {
        auto x = std::make_shared<Tensor>(X_test[i]);
        auto y_true = std::make_shared<Tensor>(y_test[i]);
        auto y_pred = model->forward(x);
        final_test_loss += mean_squared_error(y_pred, y_true)->data(0, 0);
    }
    final_test_loss /= X_test.size();
    
    std::cout << "测试集MSE损失: " << std::fixed << std::setprecision(6) 
              << final_test_loss << std::endl;
    
    // 显示一些预测样例
    std::cout << "\n预测样例 (输入 -> 真实值 -> 预测值):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(X_test.size())); ++i) {
        auto x = std::make_shared<Tensor>(X_test[i]);
        auto y_pred = model->forward(x);
        float x_val = X_test[i](0, 0);
        float true_val = y_test[i](0, 0);
        float pred_val = y_pred->data(0, 0);
        float error = std::abs(true_val - pred_val);
        
        std::cout << "  x=" << std::setw(6) << std::setprecision(3) << x_val
                  << "  y_true=" << std::setw(6) << std::setprecision(3) << true_val
                  << "  y_pred=" << std::setw(6) << std::setprecision(3) << pred_val
                  << "  error=" << std::setw(6) << std::setprecision(3) << error
                  << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MiniTensor DNN训练示例" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 示例1: XOR分类任务
    train_xor_classifier();
    
    // 示例2: 回归任务
    train_regression_model();
    
    std::cout << "\n\n所有训练示例完成！" << std::endl;
    
    return 0;
}

