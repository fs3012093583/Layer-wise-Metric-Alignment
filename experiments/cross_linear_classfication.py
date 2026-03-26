from sympy.integrals.laplace import I
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
from core.losses.multihead_prototype_loss import class_matrix_loss
from core.optimization.layerwise_optimizer import LayerwiseOptimizer, LayerwiseScheduler
from core.losses.linear_classification_loss import linear_classification_loss
  


def flatten_config(config, prefix=""):
    """递归展平配置字典，处理嵌套结构"""
    items = []
    for key, value in config.items():
        if isinstance(value, dict):
            # 处理嵌套字典
            nested_items = flatten_config(value, f"{key}_")
            items.extend(nested_items)
        else:
            # 处理基本类型
            item_key = f"{prefix}{key}"
            item_value = str(value)
            # 替换不允许的字符
            item_value = item_value.replace(".", "p").replace(",", "").replace(" ", "_")
            # 对于列表，转换为下划线连接的字符串
            if isinstance(value, list):
                item_value = "_".join(str(v) for v in value)
            items.append(f"{item_key}{item_value}")
    return items

def generate_experiment_name(config):
    """根据config的所有参数生成实验名称"""
    # 展平配置
    config_items = flatten_config(config)
    # 排序以确保一致性
    config_items.sort()
    # 限制长度，避免实验名过长
    max_length = 50
    config_str = "_".join(config_items)[:max_length]
    # 生成时间戳
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    # 组合实验名称
    return f"config_{config_str}_{current_time}"

# 在代码中使用


# =========================
# 层级损失训练函数
# =========================
def train_layerwise(model, train_loader, test_loader, monitor, config):
    """
    层级损失训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        monitor: 监控器
        config: 配置字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 损失函数
    task_criterion = nn.CrossEntropyLoss()
    # layer_criterion = class_matrix_loss
    layer_criterion = linear_classification_loss
    
    # 层级优化器
    optimizer = LayerwiseOptimizer(model, config)
    
    # 为每个层级创建可学习的权重参数
    layer_weights = {}
    all_weight_params = []
        
    is_w = False
    w = {}
    # 前向传播一次，获取各层特征维度
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        features = model(x, return_features=True)
        i = 0
        for layer_name in config["loss_weights"]:
            if layer_name != "final":
                feat = features[layer_name]
                feat = feat.view(feat.size(0), -1)
                feature_dim = feat.shape[1]
                # 创建可学习的权重参数
                if is_w == True:
                    continue
                else :
                    w[layer_name] = nn.Parameter(torch.randn(feature_dim, 100))
                    nn.init.xavier_normal_(w[layer_name])
                    i += 1
                
                w = w.to(device)
                layer_weights[layer_name] = w[layer_name]
                all_weight_params.append(w[layer_name])
        is_w = True
    
    # 为权重参数创建单独的优化器
    weight_optimizer = None
    if len(all_weight_params) > 0:
        weight_optimizer = torch.optim.Adam(all_weight_params, lr=config["lr"])
    
    # 学习率调度器
    scheduler = LayerwiseScheduler(optimizer, config)
    
    best_acc = 0.0
    global_step = 0
    
    print(f"\n{'='*60}")
    print(f"Layer-wise Training on CIFAR-100")
    print(f"Update Strategy: {config['update_strategy']}")
    print(f"Total epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Loss weights: {config['loss_weights']}")
    print(f"{'='*60}\n")
    
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_losses = {}  # 用于累积每个层级的损失
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            if weight_optimizer is not None:
                weight_optimizer.zero_grad()
            
            # 获取各层特征
            features = model(x, return_features=True)
            
            # 计算各层损失
            batch_losses = {}
            
            # 根据更新策略执行不同的损失计算和更新
            if config["update_strategy"] == "local":
                # 局部更新：为每个层级单独计算损失和更新
                # 首先计算所有层的损失，确保所有参数都有梯度
                all_losses = {}
                for layer_name, weight in config["loss_weights"].items():
                    if layer_name == "final":
                        # 最终层使用交叉熵损失
                        loss = task_criterion(features[layer_name], y)
                    else:
                        # 中间层使用自定义损失
                        # 需要将特征展平
                        feat = features[layer_name]
                        feat = feat.view(feat.size(0), -1)
                        # 传入可学习的权重参数
                        loss = layer_criterion(feat, y, num_classes=100, tau=config.get("tau", 0.1), w=layer_weights[layer_name])
                    
                    all_losses[layer_name] = weight * loss
                    batch_losses[layer_name] = loss.item()
                
                # 计算总损失并反向传播，确保所有参数都有梯度
                total_loss = sum(all_losses.values())
                total_loss.backward()
                
                # 对每个选中的层级执行参数更新
                for layer_name in config["loss_weights"].keys():
                    # 梯度裁剪（可选）
                    if config.get("clip_grad", 0) > 0:
                        # 只裁剪当前层的参数梯度
                        if hasattr(optimizer, 'layer_params') and layer_name in optimizer.layer_params:
                            params = optimizer.layer_params[layer_name]
                            torch.nn.utils.clip_grad_norm_(params, config["clip_grad"])
                    
                    # 更新当前层的参数
                    optimizer.step(layer_name)
                    
                    # 清除当前层的梯度，避免影响其他层的更新
                    if hasattr(optimizer, 'layer_params') and layer_name in optimizer.layer_params:
                        for param in optimizer.layer_params[layer_name]:
                            if param.grad is not None:
                                param.grad.zero_()
                
                # 更新权重参数
                if weight_optimizer is not None:
                    weight_optimizer.step()
            else:
                # 全局更新：一次性更新所有参数
                total_loss = 0.0
                for layer_name, weight in config["loss_weights"].items():
                    if layer_name == "final":
                        # 最终层使用交叉熵损失
                        loss = task_criterion(features[layer_name], y)
                    else:
                        # 中间层使用自定义损失
                        # 需要将特征展平
                        feat = features[layer_name]
                        feat = feat.view(feat.size(0), -1)
                        # 传入可学习的权重参数
                        loss = layer_criterion(feat, y, num_classes=100, tau=config.get("tau", 1), w=layer_weights[layer_name])
                    
                    batch_losses[layer_name] = loss.item()
                    total_loss += weight * loss
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪（可选）
                if config.get("clip_grad", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                
                # 执行优化步骤
                optimizer.step()
                if weight_optimizer is not None:
                    weight_optimizer.step()
            
            # 累积各层损失
            for layer_name, loss_value in batch_losses.items():
                if layer_name not in epoch_losses:
                    epoch_losses[layer_name] = 0.0
                epoch_losses[layer_name] += loss_value
            
            train_loss += total_loss.item()
            _, predicted = features["final"].max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            
            # 记录每个batch的loss
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
        test_acc, test_loss = evaluate_layerwise(model, test_loader, task_criterion, device)
        
        # 保存最佳模型
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), f"best_layerwise_{config['update_strategy']}.pth")
        #     print(f"  -> Best model saved! (Acc: {best_acc:.2f}%)")
        
        # 打印进度
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
        
        # 记录各层损失
        for layer_name, loss_value in epoch_losses.items():
            metrics[f"loss_{layer_name}"] = loss_value
        
        monitor.log_metrics(metrics, step=epoch)
    
    print(f"\n{'='*60}")
    print(f"Layer-wise Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    return model, best_acc

def evaluate_layerwise(model, test_loader, criterion, device):
    """
    层级模型的评估函数
    """
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

# =========================
# 运行实验
# =========================
def run_layerwise_experiments():
    """
    运行层级损失实验
    """
    # 基础配置
    base_config = {
        "batch_size": 256,
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "clip_grad": 1.0,
        "dataset": "CIFAR100",
        "model": "ResNet20",
        "optimizer": "Adam",
        "tau": 0.1
    }
    
    # 实验配置列表
    experiments = [
        {
            "name": "Layerwise_Global_Update",
            "update_strategy": "global",
            "loss_weights": {
                "layer1": 0.1,
                "layer2": 0.2,
                "layer3": 0.3,
                "final": 0.4
            }
        },
        {
            "name": "Layerwise_Local_Update",
            "update_strategy": "local",
            "loss_weights": {
                "layer1": 0.1,
                "layer2": 0.2,
                "layer3": 0.3,
                "final": 0.4
            }
        }
    ]
    
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
    
    results = {}
    
    for exp_config in experiments:
        print(f"\n{'#'*60}")
        print(f"Experiment: {exp_config['name']}")
        print(f"{'#'*60}")
        
        # 更新配置
        config = base_config.copy()
        config.update(exp_config)
        
        # 创建数据加载器
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], 
                                 shuffle=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], 
                                shuffle=False, num_workers=4)
        
        # 初始化模型
        model = resnet20_cifar()
        
        # 初始化权重
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        model.apply(init_weights)
        
        # 初始化SwanLab监控
        monitor = SwanlabMonitor(experiment_name=exp_config['name'])
        monitor.init_experiment(config=config)
        
        # 训练
        try:
            trained_model, best_acc = train_layerwise(model, train_loader, test_loader, monitor, config)
            results[exp_config['name']] = best_acc
            
        except Exception as e:
            print(f"Experiment {exp_config['name']} failed: {e}")
            results[exp_config['name']] = 0.0
        finally:
            monitor.finish()
    
    # 打印所有实验结果
    print(f"\n{'='*60}")
    print("Experiment Results Summary")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")
    
    return results

# =========================
# 简化版训练
# =========================
def train_layerwise_simple():
    """
    简化版层级损失训练
    """
    config = {
        "batch_size": 256,
        "epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "clip_grad": 1.0,
        "optimizer": "Adam",
        "tau": 1,
        "update_strategy": "global",
        "num_heads": 8,
        "loss_weights": {
            # "layer1": 0.2,
            "layer2": 0.2,
            # "layer3": 0.3,
            "final": 0.3
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
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = generate_experiment_name(config)
    monitor = SwanlabMonitor(project="GMA-Metric-Alignment-ResNet20", experiment_name=experiment_name)
    monitor.init_experiment(config=config)
    
    # 训练
    trained_model, best_acc = train_layerwise(model, train_loader, test_loader, monitor, config)
    
    print(f"\nLayerwise Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    return trained_model, best_acc

if __name__ == "__main__":
    # 运行简化版训练
    model, acc = train_layerwise_simple()
    print("交叉商损失")
    # 如果需要运行多个实验，取消下面的注释
    # results = run_layerwise_experiments()
    