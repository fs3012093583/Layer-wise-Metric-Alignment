import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sl.swanlab_init import SwanlabMonitor
from core.models.resnet import resnet20_cifar
from core.optimization.layerwise_optimizer import LayerwiseOptimizer, LayerwiseScheduler
from core.losses.linear_classification_loss import linear_classification_loss


def flatten_config(config, prefix=""):
    """递归展平配置字典，处理嵌套结构"""
    items = []
    for key, value in config.items():
        if isinstance(value, dict):
            nested_items = flatten_config(value, f"{key}_")
            items.extend(nested_items)
        else:
            item_key = f"{prefix}{key}"
            item_value = str(value)
            item_value = item_value.replace(".", "p").replace(",", "").replace(" ", "_")
            if isinstance(value, list):
                item_value = "_".join(str(v) for v in value)
            items.append(f"{item_key}{item_value}")
    return items


def generate_experiment_name(config):
    """根据config的所有参数生成实验名称"""
    config_items = flatten_config(config)
    config_items.sort()
    max_length = 50
    config_str = "_".join(config_items)[:max_length]
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    return f"config_{config_str}_{current_time}"


def train_layerwise(model, train_loader, test_loader, monitor, config):
    """
    层级损失训练函数

    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        monitor: 监控器
        config: 配置字典

    Returns:
        训练后的模型和最佳准确率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数
    task_criterion = nn.CrossEntropyLoss()

    # ========================================
    # 关键1: 为每个层级创建可学习的权重参数
    # 只在训练开始时创建一次，每个层级有独立的 w
    # ========================================
    layer_weights = {}
    all_params = []  # 存储所有需要优化的参数

    # 前向传播一次，获取各层特征维度
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        features = model(x, return_features=True)

        for layer_name in config["loss_weights"]:
            if layer_name != "final":
                feat = features[layer_name]
                # 展平特征 [B, C, H, W] -> [B, feature_dim]
                feat = feat.view(feat.size(0), -1)
                feature_dim = feat.shape[1]

                # 为这个层级创建独立的 w
                w = nn.Parameter(torch.randn(feature_dim, 100))
                nn.init.xavier_normal_(w)
                w = w.to(device)
                layer_weights[layer_name] = w
                all_params.append(w)  # 添加到参数列表

    # ========================================
    # 关键2: 模型参数和 w 使用同一个优化器
    # ========================================
    optimizer = optim.Adam(
        list(model.parameters()) + all_params,
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc = 0.0
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Layer-wise Linear Classification Training on CIFAR-100")
    print(f"Update Strategy: {config['update_strategy']}")
    print(f"Total epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Loss weights: {config['loss_weights']}")
    print(f"Layer weights: {len(layer_weights)} layers")
    print(f"{'='*60}\n")

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_losses = {}

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # ========================================
            # 关键3: 每次迭代前清零梯度
            # ========================================
            optimizer.zero_grad()

            # 获取各层特征
            features = model(x, return_features=True)

            # 计算各层损失
            total_loss = 0.0
            batch_losses = {}

            for layer_name, weight in config["loss_weights"].items():
                if layer_name == "final":
                    # 最终层使用交叉熵损失
                    loss = task_criterion(features[layer_name], y)
                else:
                    # 中间层使用线性分类损失
                    feat = features[layer_name]
                    feat = feat.view(feat.size(0), -1)
                    # ========================================
                    # 关键4: 传入该层级独立的 w
                    # ========================================
                    loss = linear_classification_loss(
                        feat, y,
                        w=layer_weights[layer_name],
                        num_classes=100,
                        tau=config.get("tau", 1.0)
                    )

                batch_losses[layer_name] = loss.item()
                total_loss += weight * loss

            # ========================================
            # 关键5: 反向传播，更新模型参数和 w
            # ========================================
            total_loss.backward()

            # 梯度裁剪（可选）
            if config.get("clip_grad", 0) > 0:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], config["clip_grad"])

            # 执行优化步骤
            optimizer.step()

            # 累积各层损失
            for layer_name, loss_value in batch_losses.items():
                if layer_name not in epoch_losses:
                    epoch_losses[layer_name] = 0.0
                epoch_losses[layer_name] += loss_value

            train_loss += total_loss.item()
            _, predicted = features["final"].max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

            if batch_idx % 50 == 0:
                monitor.log_metrics({"batch_loss": total_loss.item()}, step=global_step)
            global_step += 1

        # 计算各层的平均损失
        for layer_name in epoch_losses:
            epoch_losses[layer_name] /= len(train_loader)

        # 学习率更新
        scheduler.step()

        # 计算训练集准确率
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 测试阶段
        test_acc, test_loss = evaluate(model, test_loader, task_criterion, device)

        print(f"Epoch [{epoch+1}/{config['epochs']}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 记录到SwanLab
        metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc / 100.0,
            "test_loss": test_loss,
            "test_acc": test_acc / 100.0,
            "learning_rate": scheduler.get_last_lr()[0]
        }

        for layer_name, loss_value in epoch_losses.items():
            metrics[f"loss_{layer_name}"] = loss_value

        monitor.log_metrics(metrics, step=epoch)

    print(f"\n{'='*60}")
    print(f"Layer-wise Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

    return model, best_acc


def evaluate(model, test_loader, criterion, device):
    """评估函数"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)

    return accuracy, avg_loss


def train_layerwise_simple():
    """
    简化版层级损失训练
    """
    config = {
        "batch_size": 256,
        "epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_grad": 1.0,
        "optimizer": "Adam",
        "tau": 1.0,
        "update_strategy": "global",
        "loss_weights": {
            "layer1": 0,
            "layer2": 0,
            "layer3": 0,
            "final": 0.2
        }
    }

    # 数据加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # 初始化模型
    model = resnet20_cifar()

    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    model.apply(init_weights)

    # SwanLab监控
    experiment_name = generate_experiment_name(config)
    monitor = SwanlabMonitor(project="GMA-Metric-Alignment-ResNet20", experiment_name=experiment_name)
    monitor.init_experiment(config=config)

    # 训练
    trained_model, best_acc = train_layerwise(model, train_loader, test_loader, monitor, config)

    print(f"\nLayerwise Linear Classification Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

    return trained_model, best_acc


if __name__ == "__main__":
    model, acc = train_layerwise_simple()
    print("Linear Classification Loss Experiment")