import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from core.core_algorithm import gma_loss, compute_prototypes, evaluate
from sl.swanlab_init import SwanlabMonitor
# =========================
# 1. ViT 动态切片包装类
# =========================

class ViTEmbedSegment(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.patch_embed = m.patch_embed
        self.cls_token = m.cls_token
        self.pos_embed = m.pos_embed
        self.pos_drop = m.pos_drop
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

class ViTFinalSegment(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.last_block = m.blocks[-1]
        self.norm = m.norm
        self.head = m.head
    def forward(self, x):
        x = self.last_block(x)
        x = self.norm(x)
        # 提取 CLS Token (索引 0) 进行分类
        return self.head(x[:, 0])

def get_vit_segments(model):
    segments = nn.ModuleList()
    # Segment 0: Embedding
    segments.append(ViTEmbedSegment(model))
    # Segments 1 to N-1: Transformer Blocks (除最后一个)
    for i in range(len(model.blocks) - 1):
        segments.append(model.blocks[i])
    # Segment N: Last Block + Head (用于最终 BP)
    segments.append(ViTFinalSegment(model))
    return segments

# =========================
# 2. 混合训练主函数
# =========================

def train_vit_hybrid(model, train_loader, test_loader, monitor, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segments = get_vit_segments(model).to(device)
    num_segments = len(segments)
    
    # 按照要求：前 N-1 段用 GMA，最后一段 (Block+Head) 用 BP
    gma_limit = num_segments - 1 
    global_step = 0

    # --- 阶段一：GMA 局部对齐 ---
    for i in range(gma_limit):
        print(f"\n[GMA Phase] Training Segment {i}/{gma_limit-1}")
        optimizer = optim.Adam(segments[i].parameters(), lr=config["lr"])
        
        for epoch in range(config["epochs_per_block"]):
            segments[i].train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                with torch.no_grad():
                    feat = x
                    for j in range(i):
                        feat = segments[j](feat)
                
                # 局部训练
                feat = segments[i](feat)
                
                # ViT 对齐核心：提取 CLS Token [B, 0, D]
                alignment_feat = feat[:, 0, :]
                # 必须归一化：防止 Transformer 的量级漂移
                alignment_feat = F.normalize(alignment_feat, p=2, dim=-1)
                
                loss = gma_loss(alignment_feat, y, tau=config["tau"])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 评估该段的线性可分性 (使用原型法)
            # 注意：此处需编写适配 ViT 3D 输出的 evaluate 函数
            # acc = evaluate_vit_local(segments[:i+1], test_loader, device)
            print(f"Seg {i} Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")
            global_step += 1

    # --- 阶段二：最后一段全局 BP ---
    print(f"\n[BP Phase] Training Last Block & Head...")
    for i in range(gma_limit):
        for p in segments[i].parameters(): p.requires_grad = False
    
    final_seg = segments[-1]
    optimizer = optim.Adam(final_seg.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["global_epochs"]):
        final_seg.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = x
                for i in range(gma_limit):
                    feat = segments[i](feat)
            
            output = final_seg(feat)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 最终全局准确率评估 (argmax)
        # acc = evaluate_vit_global(segments, test_loader, device)
        print(f"BP Epoch {epoch} | Final Loss: {total_loss/len(train_loader):.4f}")

# =========================
# 3. 实验入口
# =========================

if __name__ == "__main__":
    config = {
        "batch_size": 128,
        "lr": 5e-4,
        "epochs_per_block": 5,
        "global_epochs": 20,
        "tau": 1,
    }

    # 使用 timm 构建 ViT。注意：当前 timm 版本中不存在 vit_tiny_patch4_32。
    try:
        import timm
    except ImportError:
        print("timm 未安装，请在当前解释器安装：D:/Annaconda/envs/pt-3.9/python.exe -m pip install timm")
        raise SystemExit(1)

    model_name = "vit_tiny_patch16_224"
    try:
        # 对 CIFAR-100 的 32x32 输入，显式覆盖 img_size 避免尺寸不匹配。
        model = timm.create_model(model_name, pretrained=False, num_classes=100, img_size=32)
    except Exception as e:
        print(f"创建模型失败: {model_name} | 错误: {e}")
        raise SystemExit(1)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_loader = DataLoader(datasets.CIFAR100('./data', train=True, download=True, transform=transform),
                              batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.CIFAR100('./data', train=False, transform=transform),
                             batch_size=config["batch_size"])
    experiment_name = f"GMA_ViT_CIFAR100_E{config['epochs_per_block']}_T{config['tau']}"
    monitor = SwanlabMonitor(experiment_name=experiment_name)
    monitor.init_experiment(config=config)

    # 启动训练
    # monitor = SwanlabMonitor(...) # 根据你的环境初始化
    train_vit_hybrid(model, train_loader, test_loader, monitor, config)