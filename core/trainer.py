from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


@dataclass
class StageConfig:
    epochs: int
    freeze_prefixes: List[str] | None = None
    self_heal: bool = False


class LayerWiseTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def freeze_by_prefix(self, prefixes: Iterable[str]) -> None:
        prefixes = list(prefixes)
        for name, param in self.model.named_parameters():
            param.requires_grad = not any(name.startswith(p) for p in prefixes)

    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def _step(self, x: torch.Tensor, y: torch.Tensor, self_heal: bool = False):
        self.optimizer.zero_grad(set_to_none=True)
        logits, features = self.model(x, return_features=True)
        loss, details = self.loss_fn(logits, features, y)

        if self_heal and not torch.isfinite(loss):
            # Self-heal switch skips unstable batch updates.
            self.optimizer.zero_grad(set_to_none=True)
            return None, {"skipped": 1}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()
        return loss.detach(), details

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total = 0
        correct = 0
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / max(total, 1)
        return {"acc": acc}

    def train_epoch(self, dataloader, self_heal: bool = False) -> Dict[str, float]:
        self.model.train()
        running = {"loss_total": 0.0, "loss_cls": 0.0, "loss_proto": 0.0, "loss_manifold": 0.0}
        steps = 0
        skipped = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, details = self._step(x, y, self_heal=self_heal)
            if "skipped" in details:
                skipped += 1
                continue
            for key in running:
                running[key] += details[key]
            steps += 1

        if self.scheduler is not None:
            self.scheduler.step()

        if steps == 0:
            return {"loss_total": float("nan"), "skipped": float(skipped)}

        out = {key: value / steps for key, value in running.items()}
        out["skipped"] = float(skipped)
        return out

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        self_heal: bool = False,
        checkpoint_path: str | None = None,
    ) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        best_acc = -1.0

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader, self_heal=self_heal)
            val_stats = self.evaluate(val_loader)
            row = {"epoch": float(epoch), **train_stats, "val_acc": val_stats["acc"]}
            history.append(row)

            if checkpoint_path and val_stats["acc"] > best_acc:
                best_acc = val_stats["acc"]
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)

        return history

    def fit_staged(self, train_loader, val_loader, stages: List[StageConfig]) -> List[Dict[str, float]]:
        global_history: List[Dict[str, float]] = []
        for stage_id, stage in enumerate(stages, start=1):
            if stage.freeze_prefixes:
                self.freeze_by_prefix(stage.freeze_prefixes)
            else:
                self.unfreeze_all()

            stage_history = self.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=stage.epochs,
                self_heal=stage.self_heal,
            )
            for row in stage_history:
                row["stage"] = float(stage_id)
            global_history.extend(stage_history)

        return global_history
