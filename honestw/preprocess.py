"""Splitting cells and projecting them to a low-dimensional space."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

__all__ = ["split_mask", "embed"]


def split_mask(labels, train_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask: True for training cells, chosen at random within each group."""
    labels = np.asarray(labels)
    mask = np.zeros(labels.shape[0], dtype=bool)
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        k = int(round(train_frac * idx.size))
        k = min(max(k, 1), max(idx.size - 1, 1))
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def embed(X, train: np.ndarray, n_components: int | None) -> np.ndarray:
    """Project all cells with a projection fitted on training cells only.

    Dense input uses PCA; sparse input uses truncated SVD (no centring, so the
    matrix stays sparse). With n_components=None the data are returned as is.
    """
    if n_components is None:
        return X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
    n_features = X.shape[1]
    k = int(min(n_components, n_features, int(train.sum()) - 1))
    if k < 1:
        raise ValueError("Not enough training cells or features for the projection.")
    if sparse.issparse(X):
        if k >= n_features:
            return X.toarray().astype(float)
        model = TruncatedSVD(n_components=k, random_state=0).fit(X[train])
    else:
        X = np.asarray(X, dtype=float)
        model = PCA(n_components=k, random_state=0).fit(X[train])
    return np.asarray(model.transform(X), dtype=float)
