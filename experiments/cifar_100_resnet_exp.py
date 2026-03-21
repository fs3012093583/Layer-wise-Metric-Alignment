import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sl.swanlab_init import SwanlabMonitor
from torchvision import datasets, transforms


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Linear(in_planes, out_planes) # 占位，实际用下方的 BasicBlock

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
        # 初始层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # 三个阶段的残差块
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 最后的分类层
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


import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 假设你的 core_algorithm 已经定义好了 gma_loss, compute_prototypes, evaluate
from core.core_algorithm import gma_loss, compute_prototypes, evaluate

def train_resnet_segmented(model, train_loader, test_loader, monitor, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 步骤 1: 物理拆分 ResNet 块 ---
    # 初始卷积块 + 9个残差块 + 1个全连接块
    blocks = []
    blocks.append(nn.Sequential(model.conv1, model.relu)) # Block 0
    for layer in [model.layer1, model.layer2, model.layer3]:
        for b in layer:
            blocks.append(b) # Block 1-9
    
    # 最终分类块 (包含池化和FC)
    final_stage = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)
    
    print(f"Total Blocks detected: {len(blocks)} GMA Blocks + 1 Final BP Block.")

    # --- 阶段一：前 N-1 块使用 GMA 局部训练 ---
    global_step = 0
    for b_idx in range(len(blocks)-5):
        print(f"\n[GMA Phase] Training Block {b_idx}...")
        optimizer = optim.Adam(blocks[b_idx].parameters(), lr=config["lr"])
        
        for epoch in range(config["epochs_per_block"]):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # 1. 冻结 Forward 到当前块起点
                with torch.no_grad():
                    h = x
                    for i in range(b_idx):
                        h = blocks[i](h)
                
                # 2. 局部训练当前块
                h = blocks[b_idx](h)
                
                # 3. 关键：将 4D 特征图转为向量进行对齐 (必须池化)
                pooled_h = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
                
                # 4. GMA Loss (内部务必包含 L2 Normalization)
                loss = gma_loss(pooled_h, y, tau=config["tau"])
                
                optimizer.zero_grad()
                loss.backward()
                # 无 BN 架构建议开启梯度裁剪
                torch.nn.utils.clip_grad_norm_(blocks[b_idx].parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # 评估：使用子模型计算原型准确率
            sub_model = nn.Sequential(*(blocks[:b_idx+1] + [nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()]))
            protos = compute_prototypes(sub_model, train_loader)
            acc = evaluate(sub_model, test_loader, protos)
            
            print(f"Block {b_idx} | Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc*100:.2f}%")
            monitor.log_metrics({f"B{b_idx}_loss": total_loss/len(train_loader), "test_acc": acc}, step=global_step)
            global_step += 1

    # --- 阶段二：最后一层 (FC) 进行全局 BP 微调 ---
    print(f"\n[BP Phase] Training Final Classifier...")
    # 冻结所有已经对齐好的特征块
    for i in range(len(blocks)-5):
        for p in blocks[i].parameters(): p.requires_grad = False
    
    final_optimizer = optim.Adam(final_stage.parameters(), lr=config["lr"] * 0.5)
    
    for epoch in range(config["global_epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 这里的 Forward 实际上只在计算最后一层
            with torch.no_grad():
                h = x
                for i in range(len(blocks)):
                    h = blocks[i](h)
            
            logits = final_stage(h)
            loss = F.cross_entropy(logits, y)
            
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item()
            
        # 最终全局准确率评估 (argmax)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, ty in test_loader:
                out = model(x.to(device))
                correct += (out.argmax(1) == ty.to(device)).sum().item()
                total += ty.size(0)
        
        print(f"Global BP Epoch {epoch} | Final Acc: {correct/total*100:.2f}%")
        monitor.log_metrics({"final_bp_loss": total_loss/len(train_loader), "final_acc": correct/total}, step=global_step)
        global_step += 1

    monitor.finish()

if __name__ == "__main__":
    # 1. 实验配置
    config = {
        "batch_size": 256,
        "lr": 1e-3,
        "epochs_per_block": 5,
        "global_epochs": 8,
        "tau": 1.0,         # GMA 温度参数
        "model_type": "ResNet20_NoBN",
        "dataset": "CIFAR100"
    }

    # 2. 设备准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 数据增强与加载 (适配 CIFAR-100)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    # 4. 初始化模型与权重 (无 BN 架构必须进行 Kaiming 初始化)
    model = resnet20_cifar().to(device)
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    
    model.apply(init_weights)

    # 5. SwanLab 监控初始化
    experiment_name = f"GMA_ResNet20_CIFAR100_E{config['epochs_per_block']}_T{config['tau']}"
    monitor = SwanlabMonitor(experiment_name=experiment_name)
    monitor.init_experiment(config=config)

    # 6. 执行分段混合训练
    # 注意：确保 train_resnet_segmented 函数在上方已定义
    try:
        train_resnet_segmented(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            monitor=monitor,
            config=config
        )
    except Exception as e:
        print(f"Training failed: {e}")
        monitor.finish()