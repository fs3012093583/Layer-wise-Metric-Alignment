import torch
import torch.optim as optim

class LayerwiseOptimizer:
    """
    层级优化器，支持局部更新策略
    """
    def __init__(self, model, config):
        """
        初始化层级优化器
        
        Args:
            model: 模型
            config: 配置字典
        """
        self.model = model
        self.w = w
        self.config = config
        self.update_strategy = config.get("update_strategy", "global")
        
        # 构建层级参数映射
        self.layer_params = self._build_layer_params()
        
        # 创建优化器
        if self.update_strategy == "global":
            # 全局更新：使用一个优化器更新所有参数
            if config["optimizer"] == "SGD":
                self.optimizer = optim.SGD(
                    model.parameters(),
                    lr=config["lr"],
                    momentum=config.get("momentum", 0.9),
                    weight_decay=config.get("weight_decay", 5e-4)
                )
            else:
                self.optimizer = optim.Adam(
                    model.parameters(),
                    lr=config["lr"],
                    weight_decay=config.get("weight_decay", 1e-4)
                )
        else:
            # 局部更新：为每一层创建单独的优化器
            self.optimizers = {}
            for layer_name, params in self.layer_params.items():
                if config["optimizer"] == "SGD":
                    self.optimizers[layer_name] = optim.SGD(
                        params,
                        lr=config["lr"],
                        momentum=config.get("momentum", 0.9),
                        weight_decay=config.get("weight_decay", 5e-4)
                    )
                else:
                    self.optimizers[layer_name] = optim.Adam(
                        params,
                        lr=config["lr"],
                        weight_decay=config.get("weight_decay", 1e-4)
                    )
    
    def _build_layer_params(self):
        """
        构建层级参数映射
        
        Returns:
            层级参数字典
        """
        layer_params = {}
        
        # 第一层：conv1
        layer_params["conv1"] = list(self.model.conv1.parameters())
        
        # 第二层：layer1
        layer_params["layer1"] = list(self.model.layer1.parameters())
        
        # 第三层：layer2
        layer_params["layer2"] = list(self.model.layer2.parameters())
        
        # 第四层：layer3
        layer_params["layer3"] = list(self.model.layer3.parameters())
        
        # 第五层：fc
        layer_params["final"] = list(self.model.fc.parameters())
        
        return layer_params
    
    def zero_grad(self):
        """
        清除梯度
        """
        if self.update_strategy == "global":
            self.optimizer.zero_grad()
        else:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
    
    def step(self, layer_name=None):
        """
        执行优化步骤
        
        Args:
            layer_name: 层级名称，如果为None则执行全局更新
        """
        if self.update_strategy == "global":
            self.optimizer.step()
        else:
            if layer_name and layer_name in self.optimizers:
                self.optimizers[layer_name].step()
            else:
                # 如果没有指定层级，执行所有层级的更新
                for optimizer in self.optimizers.values():
                    optimizer.step()
    
    def get_lr(self):
        """
        获取学习率
        
        Returns:
            学习率
        """
        if self.update_strategy == "global":
            return self.optimizer.param_groups[0]['lr']
        else:
            # 返回第一个优化器的学习率
            for optimizer in self.optimizers.values():
                return optimizer.param_groups[0]['lr']
            return 0.0
    
    def set_lr(self, lr):
        """
        设置学习率
        
        Args:
            lr: 学习率
        """
        if self.update_strategy == "global":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for optimizer in self.optimizers.values():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

class LayerwiseScheduler:
    """
    层级学习率调度器
    """
    def __init__(self, optimizer, config):
        """
        初始化层级学习率调度器
        
        Args:
            optimizer: 层级优化器
            config: 配置字典
        """
        self.optimizer = optimizer
        self.config = config
        
        if optimizer.update_strategy == "global":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer.optimizer, T_max=config["epochs"]
            )
        else:
            self.schedulers = {}
            for layer_name, opt in optimizer.optimizers.items():
                self.schedulers[layer_name] = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=config["epochs"]
                )
    
    def step(self):
        """
        执行学习率调度
        """
        if self.optimizer.update_strategy == "global":
            self.scheduler.step()
        else:
            for scheduler in self.schedulers.values():
                scheduler.step()
    
    def get_last_lr(self):
        """
        获取最后一次的学习率
        
        Returns:
            学习率列表
        """
        if self.optimizer.update_strategy == "global":
            return self.scheduler.get_last_lr()
        else:
            # 返回第一个调度器的学习率
            for scheduler in self.schedulers.values():
                return scheduler.get_last_lr()
            return [0.0]