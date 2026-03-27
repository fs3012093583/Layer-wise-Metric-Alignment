try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None

class SwanlabMonitor:
    def __init__(self, project="GMA-Metric-Alignment", experiment_name="Layer-wise-CIFAR100"):
        self.project = project
        self.experiment_name = experiment_name
        self.enabled = swanlab is not None
        if self.enabled:
            try:
                swanlab.login()
            except Exception:
                self.enabled = False

    def init_experiment(self, config=None):
        """
        初始化实验，传入超参数字典 config
        """
        if not self.enabled:
            return None
        experiment = swanlab.init(
            project=self.project,
            experiment_name=self.experiment_name,
            config=config,  # 记录 Batch Size, LR, Tau 等
        )
        return experiment

    @staticmethod
    def log_metrics(metrics, step=None):
        """
        记录指标，metrics 为字典，例如 {"loss": 0.1, "acc": 0.9}
        """
        if swanlab is not None:
            swanlab.log(metrics, step=step)
    
    def finish(self):
        """
        结束实验
        """
        if self.enabled:
            swanlab.finish()

# 使用示例
if __name__ == "__main__":
    # 定义超参数
    hparams = {
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "model_depth": 20,
        "model_width": 1024
    }
    
    monitor = SwanlabMonitor()
    monitor.init_experiment(config=hparams)
    
    # 在训练循环中调用
    monitor.log_metrics({"train_loss": 0.5, "test_acc": 0.12}, step=1)
