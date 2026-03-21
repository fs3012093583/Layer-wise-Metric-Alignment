import torch
import torch.nn.functional as F

def class_matrix_loss(h, y, num_classes=100, num_heads=8, tau=1):
    """
    多头原型损失函数
    
    Args:
        h: 特征向量，形状为 [B, d]
        y: 标签，形状为 [B]
        num_classes: 类别数量
        num_heads: 多头数量
        tau: 温度系数
    
    Returns:
        损失值
    """


    # 新代码
    B, d = h.shape

    # 确保 d 能被 num_heads 整除
    if d % num_heads != 0:
        # 调整 num_heads 为能整除 d 的最大因数
        while d % num_heads != 0 and num_heads > 1:
            num_heads -= 1

    d_k = d // num_heads
    h_multi = h.view(B, num_heads, d_k)
    
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