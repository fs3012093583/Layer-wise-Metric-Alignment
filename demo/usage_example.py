import torch
from core import create_model_cifar, train_flexible_sliding_window_with_final_classifier, train_full_model_with_classifier
from core import train_gma_blocks
from utils import get_cifar100_loaders

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载数据
train_loader, test_loader = get_cifar100_loaders(batch_size=256)
print("Data loaders created successfully")

# 2. 示例1：使用滑动窗口训练（多头原型损失 + 标准分类头）
print("\n=== Example 1: Sliding Window Training ===")
model = create_model_cifar(depth=8, width=1024, with_classifier=False)
model.to(device)

trained_model, classifier = train_flexible_sliding_window_with_final_classifier(
    model, train_loader, test_loader, 
    window_size=2, stride=1, epochs_per_step=3
)

# 可选：完整微调
# trained_model, classifier, best_acc = train_full_model_with_classifier(
#     trained_model, classifier, train_loader, test_loader, epochs=10
# )

# 3. 示例2：使用GMA块训练
print("\n=== Example 2: GMA Blocks Training ===")
gma_model = create_model_cifar(depth=5, width=1024, with_classifier=False)
gma_model.to(device)

history = train_gma_blocks(gma_model, train_loader, test_loader, epochs_per_layer=3, num_classes=100)

print("\n=== Training Examples Complete ===")