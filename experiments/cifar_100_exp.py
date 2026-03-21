import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.core_algorithm import gma_loss, compute_prototypes, evaluate
from sl.swanlab_init import SwanlabMonitor

# =========================
def create_model_cifar(depth=20, width=1024, input_dim=3*32*32):
    layers = [nn.Flatten()]
    for i in range(depth):
        in_dim = input_dim if i == 0 else width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, width))  # 最后一层输出维度保持宽度，后续通过原型分类
    return nn.Sequential(*layers).to(device)



def train_layerwise(model, train_loader, test_loader, monitor, epochs_per_layer=3):
    linear_layers = [i for i, layer in enumerate(model) if isinstance(layer, nn.Linear)]
    
    idx = 0
    for layer_idx in linear_layers:
        print(f"\n--- Training Layer {layer_idx} ---")
        optimizer = torch.optim.Adam(model[layer_idx].parameters(), lr=1e-3)
        

        for epoch in range(epochs_per_layer):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                # 冻结前面的层
                with torch.no_grad():
                    h = x
                    for i in range(layer_idx):
                        h = model[i](h)
                
                # 训练当前层
                h = model[layer_idx](h)
                if layer_idx + 1 < len(model) and isinstance(model[layer_idx+1], nn.ReLU):
                    h = model[layer_idx+1](h)
                
                loss = gma_loss(h, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 每一层 Epoch 结束打印一次 Test Acc 观察“自愈”过程
            current_model = model[:layer_idx+2] # 包含当前的 Linear 和 ReLU
            protos = compute_prototypes(current_model, train_loader)
            train_acc = evaluate(current_model, train_loader, protos)
            test_acc = evaluate(current_model, test_loader, protos)
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {test_acc*100:.2f}%| Train Acc: {train_acc*100:.2f}%")
            monitor.log_metrics({"train_loss": total_loss/len(train_loader), "test_acc": test_acc, "train_acc": train_acc}, step=idx*epochs_per_layer + epoch)

        idx += 1
    print("\nTraining Finished")
    monitor.finish()

if __name__ == "__main__":

    config = {
        "batch_size": 256,
        "learning_rate": 1e-3,
        "model_depth": 20,
        "model_width": 1024,
        "epochs_per_layer": 5,
        
    }

    expirement_name = f"1/CIFAR100_Layerwise_Depth{config['model_depth']}_Width{config['model_width']}"
    monitor = SwanlabMonitor(experiment_name=expirement_name)
    monitor.init_experiment(config=config)
    
    # 在训练循环中调用
    monitor.log_metrics({"train_loss": 0.5, "test_acc": 0.12}, step=1)
    # 深度设为 20，宽度设为 1024 应对 100 类
    device = "cuda" if torch.cuda.is_available() else "cpu"


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    # CIFAR-100 建议 batch_size 稍微大一点或保持 256
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = create_model_cifar(depth=config["model_depth"], width=config["model_width"])
    print("the shape of train_loader:", len(train_loader.dataset))
    train_layerwise(my_model, train_loader, test_loader,monitor, epochs_per_layer=config["epochs_per_layer"]  )