---
name: swanlab-init
description: "Use when: 需要在 PyTorch 训练中接入 SwanLab 实验追踪、统一初始化日志、记录超参数与指标、收尾结束实验、排查监控调用错误（如缺少 finish 或日志未写入）。"
---

# SwanLab Init Skill

## 目标
将 SwanLab 接入流程标准化，避免“能跑但日志不完整”的情况，确保每次实验都具备：
- 可复现实验配置（config）
- 按 step 对齐的训练指标（loss/acc/f1/lr 等）
- 明确的实验结束动作（finish）

## 何时使用
当用户提出以下诉求时使用本技能：
- “帮我接入 SwanLab”
- “训练里怎么记录指标到 SwanLab”
- “为什么 SwanLab 没有曲线/没有结束状态”
- “需要统一封装 monitor/logger”
- “修复 SwanLabMonitor 调用报错”

## 输入期望
执行本技能前，优先确认以下信息（若缺失可先用默认值）：
- 项目名 project（默认 GMA-Metric-Alignment）
- 实验名 experiment_name（建议包含模型/数据集/关键超参）
- 关键配置 config（batch_size、learning_rate、epoch、seed 等）
- 日志粒度（每 batch / 每 epoch）

## 标准执行流程
1. 检查依赖
- 确认 swanlab 可导入。
- 若不可导入，提示在当前解释器安装 swanlab。

2. 统一封装监控类
- 在项目中提供 SwanlabMonitor（或同名包装器）。
- 公开最小接口等：init_experiment(config)、log_metrics(metrics, step)、finish()。

3. 训练入口初始化
- 在训练开始前调用 init_experiment(config=config)。
- 确保 config 包含关键实验参数。

4. 训练过程记录
- 在稳定位置记录指标（推荐每 epoch 至少一次）。
- step 必须单调递增，避免曲线覆盖或倒序。

5. 结束阶段收尾
- 训练循环完成后调用 finish()。
- 如果训练可能提前退出，建议在 finally 中调用 finish()。

6. 最小验证
- 运行一次短训练或最小脚本。
- 确认 Web 端看到 config、曲线、实验结束状态。

## 参考实现模板
以下模板用于快速生成或修复监控封装。

```python
import swanlab


class SwanlabMonitor:
	def __init__(self, project="GMA-Metric-Alignment", experiment_name="exp"):
		self.project = project
		self.experiment_name = experiment_name

	def init_experiment(self, config=None):
		return swanlab.init(
			project=self.project,
			experiment_name=self.experiment_name,
			config=config,
		)

	@staticmethod
	def log_metrics(metrics, step=None):
		swanlab.log(metrics, step=step)

	@staticmethod
	def finish():
		swanlab.finish()
```

训练中调用模板：

```python
monitor = SwanlabMonitor(project="GMA-Metric-Alignment", experiment_name="vit_cifar100")
monitor.init_experiment(config=config)

global_step = 0
for epoch in range(num_epochs):
	train_loss = ...
	val_acc = ...
	monitor.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=global_step)
	global_step += 1

monitor.finish()
```

## 常见问题与修复
1. 报错：对象没有 finish
- 原因：调用方与封装类接口不一致，或导入了错误模块。
- 修复：统一封装类接口，确保存在 finish；检查导入路径唯一且正确。

2. 已记录但 Web 端无曲线
- 原因：未调用 init_experiment，或 step 未递增。
- 修复：先 init，再 log；step 使用全局递增计数。

3. 训练结束状态异常
- 原因：异常中断后未执行 finish。
- 修复：将 finish 放入 finally。

## 质量检查清单
- 可以成功导入 swanlab。
- 训练开始前已调用 init_experiment。
- 指标记录至少包含一个 loss 和一个 acc/f1。
- step 连续递增。
- 训练结束后调用 finish。

## 与本仓库结合建议
- 监控封装建议放在 sl/swanlab_init.py。
- 实验脚本统一通过 from sl.swanlab_init import SwanlabMonitor 导入，避免重复实现。
- 新实验命名建议包含数据集、模型、关键超参，便于横向对比。