import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. 数据集适配 (CIFAR-100 标准化)
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# =========================
# 2. 模型构建 (支持可选的分类头)
# =========================
def create_model_cifar(depth=20, width=1024, input_dim=3*32*32, with_classifier=False):
    layers = [nn.Flatten()]
    for i in range(depth):
        in_dim = input_dim if i == 0 else width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, width))  # 最后一层输出维度保持宽度
    
    model = nn.Sequential(*layers).to(device)
    
    # 如果需要分类头，额外添加
    if with_classifier:
        classifier = nn.Linear(width, 100).to(device)
        return model, classifier
    return model

# =========================
# 3. 多头原型损失 (仅在中间层使用)
# =========================
def class_matrix_loss(h, y, num_classes=100, num_heads=8, tau=0.1):
    B, d = h.shape
    d_k = d // num_heads
    h_multi = h.view(B, num_heads, d_k)
    
    # 子空间归一化
    h_multi = F.normalize(h_multi, p=2, dim=-1)
    
    y_onehot = F.one_hot(y, num_classes).float().to(h.device)
    count = y_onehot.sum(dim=0) + 1e-6
    
    # 计算归一化后的多头原型
    prototypes = torch.einsum('bc, bnd -> cnd', y_onehot, h_multi) / count.view(num_classes, 1, 1)
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    
    # 计算余弦相似度并缩放
    sim_per_head = torch.einsum('bnd, cnd -> bcn', h_multi, prototypes)
    sim_per_head = sim_per_head / tau
    
    # LogSumExp 聚合
    logits = torch.logsumexp(sim_per_head, dim=-1) - torch.log(torch.tensor(num_heads, dtype=torch.float))
    return F.cross_entropy(logits, y)

# =========================
# 4. 标准交叉熵损失 (最后一层使用)
# =========================
def standard_classification_loss(logits, y):
    return F.cross_entropy(logits, y)

# =========================
# 5. 原型计算 (用于多头评估)
# =========================
def compute_prototypes(model, loader, num_classes=100, num_heads=8):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model(x)
            features.append(h)
            labels.append(y)
    
    features = torch.cat(features)
    labels = torch.cat(labels).to(device)
    B_total, d = features.shape
    d_k = d // num_heads
    
    h_multi = features.view(B_total, num_heads, d_k)
    h_multi = F.normalize(h_multi, p=2, dim=-1)
    
    y_onehot = F.one_hot(labels, num_classes).float()
    count = y_onehot.sum(dim=0) + 1e-6
    protos = torch.einsum('bc, bnd -> cnd', y_onehot, h_multi) / count.view(num_classes, 1, 1)
    return F.normalize(protos, p=2, dim=-1)

# =========================
# 6. 多头评估函数
# =========================
def evaluate_multhead(model, loader, prototypes, num_heads=8, tau=0.1):
    model.eval()
    correct, total = 0, 0
    d_k = prototypes.shape[2]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h = model(x)
            h_multi = F.normalize(h.view(h.size(0), num_heads, d_k), p=2, dim=-1)
            
            sim_per_head = torch.einsum('bnd, cnd -> bcn', h_multi, prototypes)
            logits = torch.logsumexp(sim_per_head / tau, dim=-1)
            
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# =========================
# 7. 标准分类头评估函数
# =========================
def evaluate_standard(model, classifier, loader):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h = model(x)
            logits = classifier(h)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# =========================
# 8. 改进的滑动窗口训练 (最后一层切换到标准分类)
# =========================
def train_flexible_sliding_window_with_final_classifier(model, train_loader, test_loader, 
                                                        window_size=2, stride=1, epochs_per_step=3):
    """
    滑动窗口训练，最后一层切换到标准分类头进行微调
    """
    # 获取所有线性层的索引
    linear_indices = [i for i, layer in enumerate(model) if isinstance(layer, nn.Linear)]
    num_linears = len(linear_indices)
    
    # 创建分类头（用于最后阶段）
    final_width = linear_indices[-1]  # 最后一层的输出维度
    # 获取最后一层的输出维度
    for idx in reversed(linear_indices):
        if isinstance(model[idx], nn.Linear):
            final_width = model[idx].out_features
            break
    
    classifier = nn.Linear(final_width, 100).to(device)
    classifier_optimizer = None
    
    # 滑动窗口训练
    for start_idx in range(0, num_linears - window_size + 1, stride):
        current_linear_indices = linear_indices[start_idx: start_idx + window_size]
        
        # 判断是否是最后一组
        is_final_stage = (start_idx + window_size >= num_linears)
        
        print(f"\n{'='*60}")
        print(f"[Stage] Training Layers {current_linear_indices}")
        if is_final_stage:
            print("[Mode] FINAL STAGE - Using Standard Classification Loss")
        else:
            print("[Mode] Intermediate Stage - Using Multi-Head Prototype Loss")
        print(f"{'='*60}")
        
        # 配置优化器
        params_config = []
        for i, idx in enumerate(current_linear_indices):
            lr_scale = (i + 1) / window_size
            params_config.append({'params': model[idx].parameters(), 'lr': 1e-3 * lr_scale})
        
        optimizer = torch.optim.Adam(params_config)
        
        # 如果是最后阶段，初始化分类头优化器
        if is_final_stage:
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        
        for epoch in range(epochs_per_step):
            model.train()
            if is_final_stage:
                classifier.train()
            
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # 冻结窗口之前的层
                with torch.no_grad():
                    feat = x
                    for layer_idx in range(current_linear_indices[0]):
                        feat = model[layer_idx](feat)
                
                # 前向传播：穿过当前窗口
                h = feat
                last_idx = current_linear_indices[-1]
                end_flow = last_idx + 2 if (last_idx + 1 < len(model) and isinstance(model[last_idx+1], nn.ReLU)) else last_idx + 1
                
                for layer_idx in range(current_linear_indices[0], end_flow):
                    h = model[layer_idx](h)
                
                # 根据阶段选择损失函数
                if is_final_stage:
                    # 最后一层：使用标准分类头
                    logits = classifier(h)
                    loss = standard_classification_loss(logits, y)
                    
                    classifier_optimizer.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    classifier_optimizer.step()
                    optimizer.step()
                else:
                    # 中间层：使用多头原型损失
                    loss = class_matrix_loss(h, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
            
            # 评估当前进度
            current_sub_model = model[:end_flow]
            if is_final_stage:
                # 最后阶段用标准分类头评估
                test_acc = evaluate_standard(current_sub_model, classifier, test_loader)
                train_acc = evaluate_standard(current_sub_model, classifier, train_loader)
                print(f"Epoch {epoch+1}/{epochs_per_step} | Loss: {total_loss/len(train_loader):.4f} | "
                      f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            else:
                # 中间阶段用多头评估
                protos = compute_prototypes(current_sub_model, train_loader)
                test_acc = evaluate_multhead(current_sub_model, test_loader, protos)
                print(f"Epoch {epoch+1}/{epochs_per_step} | Loss: {total_loss/len(train_loader):.4f} | "
                      f"Test Acc: {test_acc*100:.2f}%")
    
    return model, classifier

# =========================
# 9. 完整的端到端训练（可选）
# =========================
def train_full_model_with_classifier(model, classifier, train_loader, test_loader, epochs=10, lr=1e-3):
    """
    完整微调阶段：联合训练 backbone 和分类头
    """
    print(f"\n{'='*60}")
    print("[Final] Full Fine-tuning with Standard Classifier")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            h = model(x)
            logits = classifier(h)
            loss = standard_classification_loss(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # 评估
        test_acc = evaluate_standard(model, classifier, test_loader)
        train_acc = evaluate_standard(model, classifier, train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
            }, 'best_model.pth')
            print(f"  -> Best model saved! (Acc: {best_acc*100:.2f}%)")
    
    return model, classifier, best_acc

if __name__ == "__main__":
    # 创建模型（不包含分类头）
    print(f"device: {device}")
    my_model = create_model_cifar(depth=8, width=1024, with_classifier=False)
    
    # 滑动窗口训练（最后一层自动切换到标准分类）
    trained_model, classifier = train_flexible_sliding_window_with_final_classifier(
        my_model, train_loader, test_loader, 
        window_size=2, stride=1, epochs_per_step=3
    )
    
    # 可选：完整微调阶段
    # trained_model, classifier, best_acc = train_full_model_with_classifier(
    #     trained_model, classifier, train_loader, test_loader, epochs=10
    # )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    # 最终评估
    final_acc = evaluate_standard(trained_model, classifier, test_loader)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")