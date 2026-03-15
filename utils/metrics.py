from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def intrinsic_dimension(features: np.ndarray, energy: float = 0.95) -> int:
    """Estimate intrinsic dimension by PCA energy threshold."""
    x = features - features.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s**2
    ratio = np.cumsum(var) / (np.sum(var) + 1e-12)
    return int(np.searchsorted(ratio, energy) + 1)


def linear_separability_score(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def manifold_contraction_rate(raw_features: np.ndarray, aligned_features: np.ndarray) -> float:
    """Pairwise distance shrink ratio (<1 means contracted manifold)."""
    raw_dist = np.mean(pdist(raw_features, metric="euclidean"))
    aligned_dist = np.mean(pdist(aligned_features, metric="euclidean"))
    return float(aligned_dist / (raw_dist + 1e-12))


def classwise_compactness(features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    intra = []
    inter_centers = []
    for cls in np.unique(labels):
        cls_feat = features[labels == cls]
        center = cls_feat.mean(axis=0)
        intra.append(np.mean(np.linalg.norm(cls_feat - center, axis=1)))
        inter_centers.append(center)

    inter_centers = np.array(inter_centers)
    inter = np.mean(pdist(inter_centers, metric="euclidean")) if len(inter_centers) > 1 else 0.0
    return float(np.mean(intra)), float(inter)
