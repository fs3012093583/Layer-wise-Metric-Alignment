import torch
import torch.nn as nn

def create_model_cifar(depth=20, width=1024, input_dim=3*32*32, with_classifier=False):
    """
    创建CIFAR数据集的MLP模型
    
    Args:
        depth: 模型深度（线性层数量）
        width: 隐藏层宽度
        input_dim: 输入维度
        with_classifier: 是否添加分类头
    
    Returns:
        模型（带或不带分类头）
    """
    layers = [nn.Flatten()]
    for i in range(depth):
        in_dim = input_dim if i == 0 else width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, width))  # 最后一层输出维度保持宽度
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(*layers).to(device)
    
    # 如果需要分类头，额外添加
    if with_classifier:
        classifier = nn.Linear(width, 100).to(device)
        return model, classifier
    return model