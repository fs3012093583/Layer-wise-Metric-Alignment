from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class UnifiedPrototypicalManifoldLoss(nn.Module):
    """
    Unified matrix-style objective:
    1) classification term
    2) prototype alignment term
    3) manifold compactness/separation term
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.1,
        margin: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cls_loss = F.cross_entropy(logits, targets)
        proto_loss = self._prototype_alignment(features, targets)
        manifold_loss = self._manifold_term(features, targets)

        total = self.alpha * cls_loss + self.beta * proto_loss + self.gamma * manifold_loss
        details = {
            "loss_total": total.item(),
            "loss_cls": cls_loss.item(),
            "loss_proto": proto_loss.item(),
            "loss_manifold": manifold_loss.item(),
        }
        return total, details

    def _prototype_alignment(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        unique_labels = torch.unique(targets)
        losses = []
        for label in unique_labels:
            mask = targets == label
            if mask.sum() <= 1:
                continue
            class_feats = features[mask]
            center = class_feats.mean(dim=0, keepdim=True)
            losses.append(((class_feats - center) ** 2).sum(dim=1).mean())

        if not losses:
            return torch.zeros((), device=features.device, dtype=features.dtype)
        return torch.stack(losses).mean()

    def _manifold_term(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = features.size(0)
        if n <= 1:
            return torch.zeros((), device=features.device, dtype=features.dtype)

        distances = torch.cdist(features, features, p=2)
        same = targets.unsqueeze(1).eq(targets.unsqueeze(0))
        same.fill_diagonal_(False)
        diff = ~same
        diff.fill_diagonal_(False)

        intra = distances[same].mean() if same.any() else torch.zeros((), device=features.device)
        inter_push = (
            F.relu(self.margin - distances[diff]).mean()
            if diff.any()
            else torch.zeros((), device=features.device)
        )
        return intra + inter_push + self.eps
