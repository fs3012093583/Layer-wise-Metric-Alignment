import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. 核心 GMA Loss (带动态锐化)
# =========================
def gma_loss(h, y, num_classes=100, tau=1,version=0):
    """
    Unified Prototypical Metric Alignment Loss
    h: [B, d], y: [B]
    tau: 温度系数，越小对齐越锐利
    version: 版本号，用于区分不同的损失函数实现
    """
    B, d = h.shape
    y_onehot = F.one_hot(y, num_classes).float().to(h.device)

    # 动态计算当前 Batch 的原型 (Prototypes)
    count = y_onehot.sum(dim=0) + 1e-6  #各个类别的数量
    prototypes = (y_onehot.T @ h) / count.unsqueeze(1) # [C, d]  #各类别样本的均值向量
    # prototypes = F.normalize(prototypes, p=2, dim=1) # 将原型长度缩放到 1
    
    # 相似度矩阵缩放(锐化处理)
    # 这一步是流形收缩的关键
    sim = (h @ prototypes.T) / tau # [B, C] h与类别中心的相似矩阵

    #当batch小的时候，bacth << C 的时候容易让聚类中心各自分开，batch >> C时候更容易是样本靠近样本中心？？？？
    
    # 隐式包含了类内凝聚和类间排斥
    loss = F.cross_entropy(sim, y)
    return loss

# =========================
# 2. 评估与原型计算
# =========================
@torch.no_grad()
def compute_prototypes(model, loader, num_classes=100):
    model.eval()
    all_h, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        h = model(x)
        all_h.append(h)
        all_y.append(y)
    
    h_cat = torch.cat(all_h)
    y_cat = torch.cat(all_y).to(device)
    
    y_onehot = F.one_hot(y_cat, num_classes).float()
    count = y_onehot.sum(dim=0) + 1e-6
    prototypes = (y_onehot.T @ h_cat) / count.unsqueeze(1)
    return prototypes

@torch.no_grad()
def evaluate(model, loader, prototypes):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h = model(x)
        sim = h @ prototypes.T
        pred = sim.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# =========================
# 3. 健壮的逐层训练框架
# =========================
# =========================
# 2. 优化后的训练逻辑
# =========================
def train_gma_blocks(model, train_loader, test_loader, epochs_per_layer=3, num_classes=100):
    # --- 第一步：解析模型结构，定义训练块 ---
    # 我们寻找 Linear 及其后的激活层作为一个 Block
    blocks = []
    current_block_start = 0
    
    for i, m in enumerate(model):
        if isinstance(m, nn.Linear):
            # 查找该 Linear 后的第一个激活层作为终点
            block_end = i
            if i + 1 < len(model) and isinstance(model[i+1], nn.ReLU):
                block_end = i + 1
            
            blocks.append({
                'id': len(blocks),
                'start': current_block_start,
                'train_layer_idx': i,  # 真正需要更新参数的 Linear 层
                'block_end': block_end # Forward 的终点
            })
            # 下一个块从当前终点之后开始
            current_block_start = block_end + 1

    print(f"Detected {len(blocks)} Training Blocks.")
    performance_history = {}

    # --- 第二步：按 Block 顺序训练 ---
    for block in blocks:
        layer_id = block['train_layer_idx']
        end_id = block['block_end']
        
        # 动态锐化系数 Tau
        tau = 1.0 - (block['id'] / len(blocks)) * (1.0 - 0.1)
        
        print(f"\n[Block {block['id']+1}/{len(blocks)}] Training Layer: {layer_id} (to End ID: {end_id}) | Tau: {tau:.4f}")
        
        # 只为当前块中的 Linear 层定义优化器
        optimizer = torch.optim.Adam(model[layer_id].parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(model[layer_id].parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

        for epoch in range(epochs_per_layer):
            model.train()
            total_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # 1. 冻结式 Forward：直到当前块的起点之前
                with torch.no_grad():
                    h = x
                    if isinstance(h, torch.Tensor) and len(h.shape) > 2:
                        h = model[0](h) # 处理 Flatten
                    
                    # 运行之前已经训练好的所有块
                    for i in range(1, block['start']):
                        h = model[i](h)

                # 2. 局部 Forward：运行当前 Block
                # 这里会经过 Linear 和 ReLU（如果有）
                for i in range(block['start'], end_id + 1):
                    h = model[i](h)

                # 3. 计算对齐 Loss
                loss = gma_loss(h, y, num_classes=num_classes, tau=tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 评估：使用从开头到当前块终点的子模型
            sub_model = model[:end_id + 1]
            protos = compute_prototypes(model, train_loader, num_classes)
            test_acc = evaluate(model, test_loader, protos)
            
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {test_acc*100:.2f}%")
            performance_history[f"B{block['id']}_E{epoch}"] = test_acc

    return performance_history
# =========================
# 4. 主程序 (以 CIFAR-100 为例)
# =========================
if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_set = CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_set = CIFAR100(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)

    # 模型定义 (5层 1024 宽)
    depth, width = 5, 1024
    layers = [nn.Flatten()]
    input_dim = 3*32*32
    for i in range(depth):
        layers.append(nn.Linear(input_dim if i == 0 else width, width))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(width, width))  # 输出层
    model = nn.Sequential(*layers).to(device)

    # 开始训练
    history = train_gma_blocks(model, train_loader, test_loader, epochs_per_layer=3, num_classes=100)
    
    print("\nTraining complete. Final Results Logged.")