import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# dataset (适配 CIFAR-100)
# =========================
# CIFAR-100 是彩色图 (3通道)，且需要一定的标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# =========================
# model (适配 3*32*32 输入和 100 分类)
# =========================
def create_model_bp_cifar(ddeepth=20, width=512, input_dim=3*32*32):
    layers = [nn.Flatten()]
    for _ in range(ddeepth):
        # 增加宽度以应对 CIFAR-100 的复杂度，建议 width 设为 512 或 1024
        layers.append(nn.Linear(input_dim if _ == 0 else width, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 100)) # 100 个类别
    return nn.Sequential(*layers).to(device)

def evaluate_bp(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def train_global_bp(model, train_loader, test_loader, epochs=30):
    print(f"Starting Global BP Training (CIFAR-100) on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        test_acc = evaluate_bp(model, test_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch} | Loss: {avg_loss:.6f} | Test Acc: {test_acc*100:.2f}% | Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # 注意：在 CIFAR-100 上，20层 MLP 且不带 Residual 的 BP 模型极难训练
    model = create_model_bp_cifar(ddeepth=5, width=1024)
    train_global_bp(model, train_loader, test_loader, epochs=30)