import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader

# 确保导入路径正确
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.core_algorithm import gma_loss, compute_prototypes, evaluate
from sl.swanlab_init import SwanlabMonitor

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. 模型定义 (保持无 BN 结构以确保横向对比公平)
# =========================
def create_model_cifar(depth=10, width=1024, input_dim=3*32*32):
    layers = [nn.Flatten()]
    for i in range(depth):
        in_dim = input_dim if i == 0 else width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers).to(device)

# =========================
# 2. 核心训练逻辑：分段 GMA + 全局 BP
# =========================
def train_hybrid(model, train_loader, test_loader, monitor, config):
    # 自动识别所有 Linear 层
    linear_indices = [i for i, layer in enumerate(model) if isinstance(layer, nn.Linear)]
    num_layers = len(linear_indices)
    mid_idx = num_layers // 2  
    
    # 划分阶段
    gma_indices = linear_indices[:mid_idx]      
    bp_start_idx = linear_indices[mid_idx] 
    
    # 将 GMA 阶段进一步划分为 Block (每 2 层一个 Block 效果更稳)
    block_size = 1
    gma_blocks = [gma_indices[i:i + block_size] for i in range(0, len(gma_indices), block_size)]

    print(f"Total Layers: {num_layers} | GMA Blocks: {len(gma_blocks)} | BP Start Layer: {bp_start_idx}")

    global_step = 0

    # --- 阶段一：GMA 分块局部优化 ---
    for b_idx, block_layers in enumerate(gma_blocks):
        print(f"\n[GMA Phase] --- Training Block {b_idx+1} (Layers {block_layers}) ---")
        
        # 优化器包含当前块内所有 Linear 层
        params = []
        for l_idx in block_layers:
            params.extend(list(model[l_idx].parameters()))
        optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
        
        # 确定 Block 的前向传播终点（包含 ReLU）
        block_end = block_layers[-1]
        if block_end + 1 < len(model) and isinstance(model[block_end+1], nn.ReLU):
            block_end += 1

        for epoch in range(config["epochs_per_layer"]):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # 1. 冻结 Forward 到 Block 起点
                with torch.no_grad():
                    h = model[:block_layers[0]](x)
                
                # 2. 局部 Forward (当前 Block 内部)
                h = model[block_layers[0] : block_end + 1](h)
                
                # 3. 计算对齐 Loss (注意：gma_loss 内部应包含 h = F.normalize(h, p=2, dim=1))
                loss = gma_loss(h, y, tau=1)
                
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：无 BN 架构的关键，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            # 评估当前 Block 的表征能力
            sub_model = model[:block_end + 1]
            protos = compute_prototypes(sub_model, train_loader)
            test_acc = evaluate(sub_model, test_loader, protos)
            
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {test_acc*100:.2f}%")
            monitor.log_metrics({"gma_loss": total_loss/len(train_loader), "test_acc": test_acc}, step=global_step)
            global_step += 1

    # --- 阶段二：全局 BP 优化后一半 ---
    print(f"\n[BP Phase] --- Global BP from Layer {bp_start_idx} ---")
    
    # 冻结前一半 (GMA 训练过的层)
    for i in range(bp_start_idx):
        for param in model[i].parameters():
            param.requires_grad = False
            
    # 获取后一半参数
    bp_params = []
    for i in range(bp_start_idx, len(model)):
        bp_params.extend(list(model[i].parameters()))
    
    # 定义分类头
    last_linear = next((m for m in reversed(model) if isinstance(m, nn.Linear)), None)
    classifier = nn.Linear(last_linear.out_features, 100).to(device)
    
    # BP 阶段的学习率建议稍微调低一点
    optimizer = torch.optim.Adam(list(bp_params) + list(classifier.parameters()), lr=config["learning_rate"] * 0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["global_epochs"]):
        model.train()
        classifier.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward
            feat = model(x)
            output = classifier(feat)
            
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bp_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 全局评估
        model.eval()
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = classifier(model(x))
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        test_acc = correct / total
        print(f"BP Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Final Acc: {test_acc*100:.2f}%")
        monitor.log_metrics({"bp_loss": total_loss/len(train_loader), "final_acc": test_acc}, step=global_step)
        global_step += 1

    monitor.finish()

# =========================
# 3. 主程序
# =========================
if __name__ == "__main__":
    config = {
        "batch_size": 256,
        "learning_rate": 1e-3,
        "model_depth": 10,
        "model_width": 1024,
        "epochs_per_layer": 5,
        "global_epochs": 20,
    }

    # Swanlab 初始化
    exp_name = f"GMA_BP_Hybrid_D{config['model_depth']}_W{config['model_width']}"
    monitor = SwanlabMonitor(experiment_name=exp_name)
    monitor.init_experiment(config=config)

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_loader = DataLoader(CIFAR100(root="./data", train=True, download=True, transform=transform), 
                              batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(CIFAR100(root="./data", train=False, download=True, transform=transform), 
                             batch_size=config["batch_size"])

    # 创建模型并执行混合训练
    my_model = create_model_cifar(depth=config["model_depth"], width=config["model_width"])
    train_hybrid(my_model, train_loader, test_loader, monitor, config)