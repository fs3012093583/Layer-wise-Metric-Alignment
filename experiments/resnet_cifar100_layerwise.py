import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover - optional dependency
    datasets = None
    transforms = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sl.swanlab_init import SwanlabMonitor
from core.models.resnet import resnet20_cifar

  

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


class LocalCIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        split = "train" if train else "test"
        file_path = Path(root) / "cifar-100-python" / split
        with open(file_path, "rb") as f:
            raw = pickle.load(f, encoding="bytes")
        self.data = torch.from_numpy(np.asarray(raw[b"data"]).reshape(-1, 3, 32, 32))
        self.labels = raw[b"fine_labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.float().div(255.0)
        return image, label


class CIFARTransform:
    def __init__(self, train=True):
        self.train = train
        self.mean = torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1)
        self.std = torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)

    def __call__(self, image):
        image = image.float().div(255.0)
        if self.train:
            image = nn.functional.pad(image, (4, 4, 4, 4), mode="reflect")
            top = torch.randint(0, 9, ()).item()
            left = torch.randint(0, 9, ()).item()
            image = image[:, top:top + 32, left:left + 32]
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
        return (image - self.mean) / self.std


def build_cifar100_datasets():
    if datasets is not None and transforms is not None:
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
        return train_set, test_set

    train_set = LocalCIFAR100(root="./data", train=True, transform=CIFARTransform(train=True))
    test_set = LocalCIFAR100(root="./data", train=False, transform=CIFARTransform(train=False))
    return train_set, test_set


class LayerwiseProjectionHeads(nn.Module):
    def __init__(self, feature_dims, projection_dim):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_dim, projection_dim),
            )
            for name, dim in feature_dims.items()
        })

    def forward(self, layer_name, x):
        return self.heads[layer_name](x)


def prototype_alignment_loss(h, y, num_classes=100, tau=0.2):
    h = nn.functional.normalize(h, p=2, dim=1)
    y_onehot = nn.functional.one_hot(y, num_classes).float().to(h.device)
    count = y_onehot.sum(dim=0).clamp_min(1.0)
    prototypes = (y_onehot.T @ h) / count.unsqueeze(1)
    prototypes = nn.functional.normalize(prototypes, p=2, dim=1)
    logits = (h @ prototypes.T) / tau
    return nn.functional.cross_entropy(logits, y)


def prepare_aux_feature(feature):
    if feature.dim() == 4:
        feature = nn.functional.adaptive_avg_pool2d(feature, output_size=1)
        feature = feature.flatten(1)
    else:
        feature = feature.view(feature.size(0), -1)
    return feature


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
    
    task_criterion = nn.CrossEntropyLoss()
    aux_layers = config.get("aux_layers", ["layer3"])
    feature_dims = {"layer1": 16, "layer2": 32, "layer3": 64, "avgpool": 64}
    projection_heads = LayerwiseProjectionHeads(
        {layer: feature_dims[layer] for layer in aux_layers},
        projection_dim=config.get("projection_dim", 128),
    ).to(device)

    if config.get("update_strategy", "global") != "global":
        print("Local update is disabled in this optimized run; falling back to global update.")

    optimizer_name = config.get("optimizer", "Adam")
    parameter_groups = list(model.parameters()) + list(projection_heads.parameters())
    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            parameter_groups,
            lr=config["lr"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 5e-4),
        )
    else:
        optimizer = optim.Adam(
            parameter_groups,
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    max_train_batches = config.get("max_train_batches")
    max_test_batches = config.get("max_test_batches")
    
    best_acc = 0.0
    global_step = 0
    
    print(f"\n{'='*60}")
    print(f"Layer-wise Training on CIFAR-100")
    print("Update Strategy: global")
    print(f"Total epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Aux layers: {aux_layers}")
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
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # 获取各层特征
            features = model(x, return_features=True)
            
            # 计算各层损失
            batch_losses = {}
            
            total_loss = config["loss_weights"].get("final", 1.0) * task_criterion(features["final"], y)
            batch_losses["final"] = total_loss.item()

            for layer_name in aux_layers:
                feat = prepare_aux_feature(features[layer_name])
                projected = projection_heads(layer_name, feat)
                aux_loss = prototype_alignment_loss(
                    projected,
                    y,
                    num_classes=100,
                    tau=config.get("tau", 0.2),
                )
                weight = config["loss_weights"].get(layer_name, 0.0)
                total_loss = total_loss + weight * aux_loss
                batch_losses[layer_name] = aux_loss.item()

            total_loss.backward()

            if config.get("clip_grad", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(projection_heads.parameters()),
                    config["clip_grad"],
                )

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
            
            # 记录每个batch的loss
            if batch_idx % 50 == 0:
                monitor.log_metrics({"batch_loss": total_loss.item()}, step=global_step)
            global_step += 1
        
        # 计算各层的平均损失
        effective_train_batches = max(1, min(len(train_loader), max_train_batches or len(train_loader)))
        for layer_name in epoch_losses:
            epoch_losses[layer_name] /= effective_train_batches
        
        # 学习率更新
        scheduler.step()
        
        # 计算训练集准确率
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / effective_train_batches

        # 测试阶段
        test_acc, test_loss = evaluate_layerwise(
            model,
            test_loader,
            task_criterion,
            device,
            max_batches=max_test_batches,
        )

        if test_acc > best_acc:
            best_acc = test_acc
        
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

def evaluate_layerwise(model, test_loader, criterion, device, max_batches=None):
    """
    层级模型的评估函数
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
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
    
    train_set, test_set = build_cifar100_datasets()
    
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
        "batch_size": int(os.environ.get("LMA_BATCH_SIZE", 128)),
        "epochs": int(os.environ.get("LMA_EPOCHS", 3)),
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_grad": 1.0,
        "optimizer": os.environ.get("LMA_OPTIMIZER", "Adam"),
        "tau": float(os.environ.get("LMA_TAU", 0.2)),
        "update_strategy": "global",
        "projection_dim": int(os.environ.get("LMA_PROJ_DIM", 128)),
        "max_train_batches": int(os.environ["LMA_MAX_TRAIN_BATCHES"]) if "LMA_MAX_TRAIN_BATCHES" in os.environ else None,
        "max_test_batches": int(os.environ["LMA_MAX_TEST_BATCHES"]) if "LMA_MAX_TEST_BATCHES" in os.environ else None,
        "aux_layers": ["layer3", "avgpool"],
        "loss_weights": {
            "layer3": 0.05,
            "avgpool": 0.1,
            "final": 1.0
        }
    }
    
    train_set, test_set = build_cifar100_datasets()
    
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
    print("修改loss")
    # 如果需要运行多个实验，取消下面的注释
    # results = run_layerwise_experiments()
    
