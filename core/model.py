from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn


@dataclass
class ModelConfig:
    model_type: str = "mlp"
    input_dim: int = 784
    num_classes: int = 10
    hidden_dims: List[int] | None = None
    dropout: float = 0.1
    in_channels: int = 3
    conv_channels: List[int] | None = None


class DynamicMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = x.view(x.size(0), -1)
        features = self.backbone(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


class DynamicCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int],
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        conv_layers: List[nn.Module] = []
        channels = [in_channels] + conv_channels
        for i in range(len(channels) - 1):
            conv_layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
            )
            conv_layers.append(nn.BatchNorm2d(channels[i + 1]))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))

        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(conv_channels[-1], num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        logits = self.head(x)
        if return_features:
            return logits, x
        return logits


def build_model(config: Dict[str, Any] | ModelConfig) -> nn.Module:
    if isinstance(config, dict):
        config = ModelConfig(**config)

    if config.model_type.lower() == "mlp":
        hidden_dims = config.hidden_dims or [512, 256, 128]
        return DynamicMLP(
            input_dim=config.input_dim,
            hidden_dims=hidden_dims,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    if config.model_type.lower() == "cnn":
        conv_channels = config.conv_channels or [64, 128, 256]
        return DynamicCNN(
            in_channels=config.in_channels,
            conv_channels=conv_channels,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    raise ValueError(f"Unsupported model_type: {config.model_type}")
