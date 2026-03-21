import torch.nn.functional as F

def standard_classification_loss(logits, y):
    """
    标准交叉熵损失函数
    
    Args:
        logits: 模型输出的 logits，形状为 [B, C]
        y: 标签，形状为 [B]
    
    Returns:
        损失值
    """
    return F.cross_entropy(logits, y)