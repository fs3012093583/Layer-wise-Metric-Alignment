import torch
import torch.nn as nn
from core.losses.gma_loss import gma_loss
from core.metrics.evaluation import compute_prototypes, evaluate

def train_gma_blocks(model, train_loader, test_loader, epochs_per_layer=3, num_classes=100):
    """
    健壮的逐层训练框架
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs_per_layer: 每层训练的轮数
        num_classes: 类别数量
    
    Returns:
        性能历史记录
    """
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

        for epoch in range(epochs_per_layer):
            model.train()
            total_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(model.device), y.to(model.device)

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