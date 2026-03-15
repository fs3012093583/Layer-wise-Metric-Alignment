from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import psutil
import torch


@dataclass
class MemorySnapshot:
    cpu_rss_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float


class MemoryTracker:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    def reset_gpu_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> MemorySnapshot:
        cpu_rss = self.process.memory_info().rss / (1024**2)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            allocated = reserved = peak = 0.0

        return MemorySnapshot(
            cpu_rss_mb=float(cpu_rss),
            gpu_allocated_mb=float(allocated),
            gpu_reserved_mb=float(reserved),
            gpu_peak_mb=float(peak),
        )

    def snapshot_dict(self) -> Dict[str, float]:
        snap = self.snapshot()
        return {
            "cpu_rss_mb": snap.cpu_rss_mb,
            "gpu_allocated_mb": snap.gpu_allocated_mb,
            "gpu_reserved_mb": snap.gpu_reserved_mb,
            "gpu_peak_mb": snap.gpu_peak_mb,
        }
