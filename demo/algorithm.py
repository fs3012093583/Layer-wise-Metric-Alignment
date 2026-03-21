import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# dataset
# =========================

transform = transforms.ToTensor()

train_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# =========================
# model
# =========================

# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(28*28, 2000),
#     nn.ReLU(),
#     nn.Linear(2000, 2000),
#     nn.ReLU(),
#     nn.Linear(2000, 2000),
#     nn.ReLU(),
#     nn.Linear(2000, 2000),
#     nn.ReLU(),
#     nn.Linear(2000, 1000)   # embedding 输出10维
# ).to(device)


def create_model(ddeepth=5, width=2000, embedding_dim=28*28):
    layers = [nn.Flatten()]
    for _ in range(ddeepth):
        if _ == 0:  # 第一层后面不加 ReLU
            layers.append(nn.Linear(embedding_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
    layers.append(nn.Linear(width, width))
    return nn.Sequential(*layers).to(device)

# =========================
# class matrix + CE loss
# =========================

# def class_matrix_loss(h, y, num_classes=10, lambda_ce=10):

#     B, d = h.shape

#     y_onehot = F.one_hot(y, num_classes).float().to(h.device)

#     count = y_onehot.sum(dim=0) + 1e-6

#     # prototypes
#     prototypes = (y_onehot.T @ h) / count.unsqueeze(1)

#     # 类内 loss
#     proto_expand = y_onehot @ prototypes
#     intra_loss = ((h - proto_expand) ** 2).mean()

#     # classification loss
#     sim = h @ prototypes.T
#     ce_loss = F.cross_entropy(sim, y_onehot)

#     loss = ce_loss
#     return loss
def class_matrix_loss(h, y, num_classes=100, tau=1,version=0):
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
    prototypes = F.normalize(prototypes, p=2, dim=1) # 将原型长度缩放到 1

    # 相似度矩阵缩放 (锐化处理)
    # 这一步是流形收缩的关键
    sim = (h @ prototypes.T) / tau # [B, C]
    
    # 隐式包含了类内凝聚和类间排斥
    loss = F.cross_entropy(sim, y)
    return loss
# =========================
# evaluate 函数
# =========================

def evaluate(model, loader, prototypes=None, use_proto=True):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            h = model(x)

            if use_proto and prototypes is not None:
                sim = h @ prototypes.T
                pred = sim.argmax(dim=1)
            else:
                pred = h.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return acc

# =========================
# layer-wise train
# =========================

def train_layerwise(model, train_loader, device, epochs_per_layer=5, print_acc=True):
    for layer_id in range(len(model)):

        # if layer_id <=1:  # 跳过 Flatten 层
        #     continue

        if not isinstance(model[layer_id], nn.Linear):
            continue

        print(f"\nTraining Layer: {layer_id}")
        optimizer = torch.optim.Adam(model[layer_id].parameters(), lr=1e-3)

        for epoch in range(epochs_per_layer):
            total_loss = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                # forward 到当前层
                with torch.no_grad():
                    h = x
                    for i in range(layer_id):
                        h = model[i](h)

                # 当前层 forward
                h = model[layer_id](h)

                # 如果后面有 ReLU，要一起算
                if layer_id + 1 < len(model) and isinstance(model[layer_id+1], nn.ReLU):
                    h = model[layer_id+1](h)

                loss = class_matrix_loss(h, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # 打印训练集准确率
            if print_acc:
                # 用 prototypes 预测
                prototypes = compute_prototypes(model, train_loader)
                train_acc = evaluate(model, train_loader, prototypes)
                test_acc = evaluate(model, test_loader, prototypes)
                print(f"Epoch: {epoch} | Loss: {avg_loss:.6f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            else:
                print(f"Epoch: {epoch} | Loss: {avg_loss:.6f}")

    print("\nTraining Finished")

# =========================
# compute prototypes
# =========================

def compute_prototypes(model, loader):
    model.eval()
    num_classes = 10
    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model(x)
            features.append(h)
            labels.append(y.to(device))

    features = torch.cat(features)
    labels = torch.cat(labels)

    y_onehot = F.one_hot(labels, num_classes).float().to(features.device)
    count = y_onehot.sum(dim=0) + 1e-6

    prototypes = (y_onehot.T @ features) / count.unsqueeze(1)
    return prototypes

# =========================
# main
# =========================

if __name__ == "__main__":
    model = create_model(ddeepth=5, width=1000, embedding_dim=28*28)
    train_layerwise(model, train_loader, device, epochs_per_layer=3, print_acc=True)

    prototypes = compute_prototypes(model, train_loader)

    test_acc = evaluate(model, test_loader, prototypes)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")