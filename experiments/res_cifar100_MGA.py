import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sl.swanlab_init import SwanlabMonitor

# =========================
# GMA核心算法
# =========================
def gma_loss(h, y, num_classes=100, num_heads=8, tau=1.0):
    """
    GMA (Group-wise Multi-head Alignment) Loss
    h: features [B, d]
    y: labels [B]
    """
    B, d = h.shape
    d_k = d // num_heads
    
    # 确保维度可整除
    if d_k * num_heads != d:
        d_actual = (d // num_heads) * num_heads
        h = h[:, :d_actual]
        d_k = d_actual // num_heads
    
    # 多头分解
    h_multi = h.view(B, num_heads, d_k)
    
    # 子空间归一化（只关心方向）
    h_multi = F.normalize(h_multi, p=2, dim=-1)
    
    # 计算每个类别的原型
    y_onehot = F.one_hot(y, num_classes).float().to(h.device)
    count = y_onehot.sum(dim=0) + 1e-6
    
    # 动态更新原型（使用EMA）
    prototypes = torch.einsum('bc, bnd -> cnd', y_onehot, h_multi) / count.view(num_classes, 1, 1)
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    # 计算多头相似度
    sim_per_head = torch.einsum('bnd, cnd -> bcn', h_multi, prototypes)  # [B, C, H]
    
    # LogSumExp聚合
    logits = torch.logsumexp(sim_per_head / tau, dim=-1)  # [B, C]
    
    # 交叉熵损失
    loss = F.cross_entropy(logits, y)
    
    return loss

def compute_prototypes(model, loader, num_heads=8):
    """
    计算所有类别的多头原型
    """
    device = next(model.parameters()).device
    model.eval()
    
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model(x)  # 假设模型输出已经是特征向量
            features.append(h)
            labels.append(y)
    
    features = torch.cat(features)
    labels = torch.cat(labels).to(device)
    
    B, d = features.shape
    d_k = d // num_heads
    
    if d_k * num_heads != d:
        d_actual = (d // num_heads) * num_heads
        features = features[:, :d_actual]
        d_k = d_actual // num_heads
    
    h_multi = features.view(B, num_heads, d_k)
    h_multi = F.normalize(h_multi, p=2, dim=-1)
    
    y_onehot = F.one_hot(labels, num_classes=100).float()
    count = y_onehot.sum(dim=0) + 1e-6
    protos = torch.einsum('bc, bnd -> cnd', y_onehot, h_multi) / count.view(100, 1, 1)
    
    return F.normalize(protos, p=2, dim=-1)

def evaluate_gma(model, loader, prototypes, num_heads=8, tau=1.0):
    """
    GMA模型评估
    """
    device = next(model.parameters()).device
    model.eval()
    
    correct, total = 0, 0
    d_k = prototypes.shape[2]
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model(x)
            
            B, d = h.shape
            if d_k * num_heads != d:
                d_actual = (d // num_heads) * num_heads
                h = h[:, :d_actual]
            
            h_multi = h.view(B, num_heads, d_k)
            h_multi = F.normalize(h_multi, p=2, dim=-1)
            
            sim_per_head = torch.einsum('bnd, cnd -> bcn', h_multi, prototypes)
            logits = torch.logsumexp(sim_per_head / tau, dim=-1)
            
            pred = logits.argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
    
    return correct / total

# =========================
# 模型定义（与BP基线完全一致）
# =========================
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
    def __init__(self, block, num_blocks, num_classes=100, feature_dim=64):
        super(ResNet_GMA, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # GMA使用特征输出，不加分类头
        self.feature_dim = feature_dim
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)  # [B, 64]
        
        if return_features:
            return features
        return features

def resnet20_gma():
    """与BP版本完全相同的架构，只是去掉最后的分类层"""
    return ResNet_GMA(BasicBlock, [3, 3, 3])

# =========================
# GMA分段训练
# =========================
def train_gma_segmented(model, train_loader, test_loader, monitor, config):
    """
    GMA分段训练算法
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 物理拆分ResNet块
    blocks = []
    blocks.append(nn.Sequential(model.conv1, model.relu))  # Block 0
    for layer in [model.layer1, model.layer2, model.layer3]:
        for b in layer:
            blocks.append(b)  # Block 1-9
    
    # 添加池化层用于特征提取
    pool_layer = nn.Sequential(model.avgpool, nn.Flatten())
    
    print(f"\n{'='*60}")
    print(f"GMA Segmented Training on CIFAR-100")
    print(f"Total blocks: {len(blocks)}")
    print(f"Training blocks: {len(blocks)-1} (all except last)")
    print(f"Window size: {config['window_size']}")
    print(f"Epochs per block: {config['epochs_per_block']}")
    print(f"Temperature tau: {config['tau']}")
    print(f"Learning rate: {config['lr']}")
    print(f"{'='*60}\n")
    
    global_step = 0
    best_acc = 0.0
    
    # 阶段一：滑动窗口训练
    for b_idx in range(len(blocks) - config.get('skip_last', 1)):
        # 确定窗口范围
        window_start = max(0, b_idx - config['window_size'] + 1)
        window_blocks = blocks[window_start:b_idx+1]
        
        print(f"\n[GMA Stage] Training Window [{window_start}:{b_idx+1}] "
              f"(Blocks {[i for i in range(window_start, b_idx+1)]})")
        
        # 收集需要优化的参数
        params_to_optimize = []
        for blk in window_blocks:
            params_to_optimize.extend(blk.parameters())
        
        # 使用Adam优化器，与BP基线一致
        optimizer = optim.Adam(params_to_optimize, lr=config['lr'], 
                              weight_decay=config.get('weight_decay', 1e-4))
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=config['epochs_per_block'])
        
        for epoch in range(config['epochs_per_block']):
            model.train()
            total_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # 1. 冻结窗口之前的层
                with torch.no_grad():
                    h = x
                    for i in range(window_start):
                        h = blocks[i](h)
                
                # 2. 通过窗口内的层
                for i in range(window_start, b_idx + 1):
                    h = blocks[i](h)
                
                # 3. 池化得到特征向量
                features = pool_layer(h)  # [B, 64]
                
                # 4. GMA损失
                loss = gma_loss(features, y, num_classes=100, 
                               num_heads=config['num_heads'], 
                               tau=config['tau'])
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if config.get('clip_grad', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, config['clip_grad'])
                
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            
            # 评估当前阶段
            sub_model = nn.Sequential(*blocks[:b_idx+1], pool_layer)
            sub_model.eval()
            
            # 计算原型并评估
            protos = compute_prototypes(sub_model, train_loader, 
                                       num_heads=config['num_heads'])
            test_acc = evaluate_gma(sub_model, test_loader, protos,
                                   num_heads=config['num_heads'], 
                                   tau=config['tau'])
            
            avg_loss = total_loss / len(train_loader)
            print(f"Window [{window_start}:{b_idx+1}] | Epoch {epoch+1}/{config['epochs_per_block']} | "
                  f"Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # 记录到SwanLab
            monitor.log_metrics({
                f"window_{window_start}_{b_idx+1}_loss": avg_loss,
                f"window_{window_start}_{b_idx+1}_acc": test_acc,
                "current_test_acc": test_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=global_step)
            global_step += 1
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'prototypes': protos,
                    'test_acc': test_acc
                }, "best_gma_model.pth")
                print(f"  -> Best model saved! (Acc: {best_acc*100:.2f}%)")
    
    # 阶段二：可选的最后微调（使用标准分类损失）
    if config.get('final_finetune', False):
        print(f"\n[Final Phase] Fine-tuning with Standard Classification Loss")
        
        # 添加分类头
        classifier = nn.Linear(64, 100).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),
                              lr=config['lr'] * 0.1, weight_decay=1e-4)
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(config.get('finetune_epochs', 10)):
            model.train()
            classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                features = model(x)
                logits = classifier(features)
                loss = criterion(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            
            # 评估
            model.eval()
            classifier.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    features = model(x)
                    logits = classifier(features)
                    _, predicted = logits.max(1)
                    test_total += y.size(0)
                    test_correct += (predicted == y.to(device)).sum().item()
            
            train_acc = 100.0 * correct / total
            test_acc = 100.0 * test_correct / test_total
            
            print(f"Fine-tune Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
            
            monitor.log_metrics({
                "finetune_loss": total_loss/len(train_loader),
                "finetune_train_acc": train_acc/100.0,
                "finetune_test_acc": test_acc/100.0
            }, step=global_step + epoch)
            
            if test_acc/100.0 > best_acc:
                best_acc = test_acc/100.0
    
    print(f"\n{'='*60}")
    print(f"GMA Training Complete!")
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"{'='*60}")
    
    return model, best_acc

# =========================
# GMA训练主函数
# =========================
def train_gma_main():
    """
    GMA算法主训练函数，保证与BP基线公平对比
    """
    config = {
        # 数据配置
        "batch_size": 128,
        "dataset": "CIFAR100",
        "model": "ResNet20_GMA",
        
        # 训练配置
        "epochs_per_block": 5,  # 每个窗口训练epochs
        "window_size": 2,       # 滑动窗口大小
        "skip_last": 1,         # 跳过最后的块（留给池化层）
        "lr": 1e-3,            # 学习率（与BP一致）
        "weight_decay": 1e-4,   # 权重衰减（与BP一致）
        "clip_grad": 1.0,       # 梯度裁剪
        
        # GMA特定配置
        "num_heads": 8,         # 多头数量
        "tau": 1.0,             # 温度系数
        
        # 可选：最后微调
        "final_finetune": False,  # 是否最后用标准分类损失微调
        "finetune_epochs": 10,
    }
    
    print(f"\n{'#'*60}")
    print(f"GMA Algorithm Training")
    print(f"{'#'*60}")
    
    # 数据加载（与BP完全一致）
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

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, 
                                  transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, 
                                 transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型（与BP架构相同）
    model = resnet20_gma()
    
    # 初始化权重（与BP完全一致）
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    model.apply(init_weights)
    
    # SwanLab监控
    monitor = SwanlabMonitor(experiment_name=f"GMA_ResNet20_CIFAR100_W{config['window_size']}_H{config['num_heads']}")
    monitor.init_experiment(config=config)
    
    # 训练
    try:
        trained_model, best_acc = train_gma_segmented(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            monitor=monitor,
            config=config
        )
        
        print(f"\n{'='*60}")
        print(f"GMA Training Complete!")
        print(f"Final Best Test Accuracy: {best_acc*100:.2f}%")
        print(f"{'='*60}")
        
        return trained_model, best_acc
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0
    finally:
        monitor.finish()

# =========================
# 对比实验
# =========================
def compare_bp_vs_gma():
    """
    运行BP和GMA的对比实验
    """
    print("\n" + "="*60)
    print("BP vs GMA Comparison on CIFAR-100")
    print("="*60)
    
    # 这里需要导入BP基线训练函数
    # 由于BP基线在另一个文件中，这里假设我们已经运行了BP基线并得到了结果
    # 实际使用时，可以分别运行两个脚本
    
    print("\n1. Running BP Baseline...")
    print("   (Run train_bp_baseline.py separately)")
    
    print("\n2. Running GMA Algorithm...")
    print("   (This script)")
    
    print("\n3. Compare Results:")
    print("   | Method | Best Test Acc |")
    print("   |--------|---------------|")
    print("   | BP     | ~60-65%       |")
    print("   | GMA    | ?             |")
    
    # 运行GMA
    model, gma_acc = train_gma_main()
    
    print(f"\n{'='*60}")
    print(f"GMA Final Accuracy: {gma_acc*100:.2f}%")
    print(f"Expected BP Baseline: 60-65%")
    print(f"Difference: {gma_acc*100 - 62.5:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 运行GMA训练
    model, acc = train_gma_main()
    
    # 如果需要对比，取消下面的注释
    # compare_bp_vs_gma() 
    