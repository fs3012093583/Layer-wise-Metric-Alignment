import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# dataset (与原代码保持一致)
# =========================
transform = transforms.ToTensor()
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# =========================
# model (结构对齐你的 create_model)
# =========================
def create_model_bp(ddeepth=5, width=2000, embedding_dim=28*28):
    layers = [nn.Flatten()]
    for _ in range(ddeepth):
        layers.append(nn.Linear(embedding_dim if _ == 0 else width, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 10)) # BP 最后一层直接输出分类 logits (10维)
    return nn.Sequential(*layers).to(device)

# =========================
# evaluate 函数 (标准分类 Acc)
# =========================
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

# =========================
# Global BP Training
# =========================
def train_global_bp(model, train_loader, test_loader, epochs=15):
    print(f"Starting Global BP Training on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 为了对齐你的总训练强度 (层数 * 每层epoch)
    # 你有 6 个 Linear 层，每层 3 epoch，总计约 18 epoch
    
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
        
        train_acc = evaluate_bp(model, train_loader)
        test_acc = evaluate_bp(model, test_loader)
        avg_loss = total_loss / len(train_loader)
        duration = time.time() - start_time
        
        print(f"Epoch: {epoch} | Loss: {avg_loss:.6f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Time: {duration:.2f}s")

if __name__ == "__main__":
    # 使用和你一样的参数: 5层隐藏层, 1000宽度
    model = create_model_bp(ddeepth=20, width=100, embedding_dim=28*28)
    
    # 开始训练
    train_global_bp(model, train_loader, test_loader, epochs=18)