import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_classification_loss(h, y, w=None, num_classes=100, tau=1.0):
    """
    线性分类损失函数
    对特征进行线性变换后使用交叉熵损失
    
    Args:
        h: 特征 [B, feature_dim]
        y: 标签 [B]
        w: 可学习的权重参数 [feature_dim, num_classes]，如果为 None 则使用随机权重
        num_classes: 类别数量
        tau: 温度系数（未使用，保持接口兼容）
        
    Returns:
        损失值
    """
    if w is not None:
        # 使用外部传入的权重参数
        logits = h @ w  # [B, num_classes]
        return F.cross_entropy(logits, y)
    else:
        # 使用随机权重（仅用于测试）
        feature_dim = h.shape[1]
        w = torch.randn(feature_dim, num_classes, device=h.device)
        logits = h @ w
        return F.cross_entropy(logits, y)