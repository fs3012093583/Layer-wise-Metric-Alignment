import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sl.swanlab_init import SwanlabMonitor

# =========================
# 模型定义 (与你的GMA版本相同)
# =========================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_GMA(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_GMA, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet20_cifar():
    return ResNet_GMA(BasicBlock, [3, 3, 3])

# =========================
# 标准BP训练函数
# =========================
def train_standard_bp(model, train_loader, test_loader, monitor, config):
    """
    标准反向传播训练基准
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    global_step = 0
    
    print(f"\n{'='*60}")
    print(f"Standard BP Training on CIFAR-100")
    print(f"Total epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config.get('weight_decay', 1e-4)}")
    print(f"{'='*60}\n")
    
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪（可选）
            if config.get("clip_grad", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            
            # 记录每个batch的loss
            if batch_idx % 50 == 0:
                monitor.log_metrics({"batch_loss": loss.item()}, step=global_step)
            global_step += 1
        
        # 学习率更新
        scheduler.step()
        
        # 计算训练集准确率
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 测试阶段
        test_acc, test_loss = evaluate_bp(model, test_loader, criterion, device)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_bp_model.pth")
            print(f"  -> Best model saved! (Acc: {best_acc:.2f}%)")
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{config['epochs']}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 记录到SwanLab
        monitor.log_metrics({
            "train_loss": avg_train_loss,
            "train_acc": train_acc / 100.0,
            "test_loss": test_loss,
            "test_acc": test_acc / 100.0,
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=epoch)
    
    print(f"\n{'='*60}")
    print(f"Standard BP Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    return model, best_acc

def evaluate_bp(model, test_loader, criterion, device):
    """
    标准BP模型的评估函数
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
# 不同配置的实验
# =========================
def run_experiments():
    """
    运行多个配置的BP基线实验
    """
    # 基础配置
    base_config = {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_grad": 1.0,
        "dataset": "CIFAR100",
        "model": "ResNet20"
    }
    
    # 实验配置列表
    experiments = [
        {
            "name": "BP_Adam_lr1e3",
            "lr": 1e-3,
            "optimizer": "Adam",
            "epochs": 100
        },
        {
            "name": "BP_SGD_lr1e1_momentum",
            "lr": 0.1,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "epochs": 100
        },
        {
            "name": "BP_Adam_lr5e4",
            "lr": 5e-4,
            "optimizer": "Adam",
            "epochs": 100
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
            # 根据优化器类型创建
            if config["optimizer"] == "SGD":
                optimizer = optim.SGD(model.parameters(), 
                                     lr=config["lr"], 
                                     momentum=config.get("momentum", 0.9),
                                     weight_decay=config.get("weight_decay", 5e-4))
            else:
                optimizer = optim.Adam(model.parameters(), 
                                      lr=config["lr"],
                                      weight_decay=config.get("weight_decay", 1e-4))
            
            # 这里使用简化版的训练函数，直接传入optimizer
            # 或者复用上面的train_standard_bp函数（需要修改）
            # 为了简便，我们直接调用train_standard_bp但需要传入optimizer
            # 由于train_standard_bp内部创建optimizer，我们需要修改一下
            
            # 临时解决方案：创建一个包装函数
            def train_with_optimizer(model, train_loader, test_loader, monitor, config, optimizer):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
                
                best_acc = 0.0
                for epoch in range(config["epochs"]):
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        loss.backward()
                        if config.get("clip_grad", 0) > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        train_total += y.size(0)
                        train_correct += predicted.eq(y).sum().item()
                    
                    scheduler.step()
                    train_acc = 100.0 * train_correct / train_total
                    test_acc, test_loss = evaluate_bp(model, test_loader, criterion, device)
                    
                    if test_acc > best_acc:
                        best_acc = test_acc
                        torch.save(model.state_dict(), f"best_{exp_config['name']}.pth")
                    
                    print(f"Epoch [{epoch+1}/{config['epochs']}] | "
                          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
                          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
                    
                    monitor.log_metrics({
                        "train_loss": train_loss/len(train_loader),
                        "train_acc": train_acc/100.0,
                        "test_acc": test_acc/100.0,
                        "test_loss": test_loss
                    }, step=epoch)
                
                return best_acc
            
            best_acc = train_with_optimizer(model, train_loader, test_loader, monitor, config, optimizer)
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
# 简化版训练（推荐使用）
# =========================
def train_baseline_simple():
    """
    简化版训练，只跑一个配置
    """
    config = {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_grad": 1.0,
        "tau": 1.0,  # 只是为了与GMA对比
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
    monitor = SwanlabMonitor(experiment_name="BP_Baseline_CIFAR100")
    monitor.init_experiment(config=config)
    
    # 训练
    trained_model, best_acc = train_standard_bp(model, train_loader, test_loader, monitor, config)
    
    print(f"\nBaseline Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    return trained_model, best_acc

if __name__ == "__main__":
    # 运行简化版训练
    model, acc = train_baseline_simple()
    
    # 如果需要运行多个实验，取消下面的注释
    # results = run_experiments()