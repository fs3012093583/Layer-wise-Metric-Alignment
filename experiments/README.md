# 层级损失训练实验说明

## 实验设计

本实验实现了层级损失训练策略，对比了两种更新策略：

1. **全局更新策略**：每个损失更新所有参数
2. **局部更新策略**：每个损失只更新对应层级的参数

## 目录结构

```
Layer-wise-Metric-Alignment/
├── core/
│   ├── models/
│   │   └── resnet.py        # 支持层级特征提取的ResNet模型
│   ├── losses/
│   │   └── multihead_prototype_loss.py  # 自定义层级损失
│   └── optimization/
│       └── layerwise_optimizer.py  # 层级优化器
└── experiments/
    ├── resnet_cifar100_bp.py        # 标准BP训练（对比）
    └── resnet_cifar100_layerwise.py # 层级损失训练
```

## 核心组件

### 1. 层级ResNet模型
- 支持返回中间特征（layer1, layer2, layer3）
- 保留与原始ResNet相同的网络结构
- 仅添加了返回特征的功能

### 2. 层级优化器
- 支持全局更新和局部更新两种策略
- 为局部更新策略创建了层级参数映射
- 实现了层级学习率调度器

### 3. 层级损失训练
- 在关键节点（残差合并后经过激活函数的位置）放置损失
- 最终层使用常规交叉熵损失
- 中间层使用自定义的多头原型损失
- 支持损失权重的灵活配置

## 运行方法

### 1. 运行标准BP训练（对比）

```bash
cd experiments
python resnet_cifar100_bp.py
```

### 2. 运行层级损失训练

```bash
cd experiments
python resnet_cifar100_layerwise.py
```

### 3. 运行多个实验配置

修改 `resnet_cifar100_layerwise.py` 文件，取消注释以下代码：

```python
# 如果需要运行多个实验，取消下面的注释
# results = run_layerwise_experiments()
```

然后运行：

```bash
cd experiments
python resnet_cifar100_layerwise.py
```

## 实验配置

### 基础配置
- 数据集：CIFAR-100
- 模型：ResNet-20
- 批次大小：128
- 训练轮数：100
- 学习率：1e-3
- 权重衰减：1e-4

### 层级损失配置
- 损失权重：
  - layer1: 0.1
  - layer2: 0.2
  - layer3: 0.3
  - final: 0.4
- 温度系数（tau）：0.1

## 监控与评估

- 使用SwanLab进行实验监控
- 记录训练/测试损失和准确率
- 记录各层级损失值
- 保存最佳模型

## 预期结果

- 层级损失训练应该比标准BP训练更稳定
- 全局更新策略可能在性能上更优
- 局部更新策略可能在计算效率上更优

## 扩展建议

1. 尝试不同的损失权重配置
2. 尝试不同的层级损失函数
3. 尝试在更深的网络上应用此策略
4. 尝试不同的更新策略组合（例如混合策略）