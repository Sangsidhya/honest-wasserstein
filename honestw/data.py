"""Loading data from CSV or AnnData (.h5ad) files, and a simulated screen."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_csv", "load_h5ad", "simulate_screen"]


def load_csv(path_or_buffer, label_column: str, feature_columns=None):
    """Read a table with one row for each cell.

    Returns (X, labels, feature_names). If feature_columns is None, every
    numeric column other than the label column is used.
    """
    df = pd.read_csv(path_or_buffer)
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found.")
    if feature_columns is None:
        feature_columns = [c for c in df.select_dtypes("number").columns if c != label_column]
    X = df[feature_columns].to_numpy(dtype=float)
    return X, df[label_column].astype(str).to_numpy(), list(feature_columns)


def load_h5ad(path, label_column: str, layer: str | None = None):
    """Read an AnnData file. Returns (X, labels, feature_names).

    Uses .X, or the named layer. Expression should already be normalised
    (e.g. scanpy.pp.normalize_total followed by scanpy.pp.log1p).
    """
    try:
        import anndata
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading .h5ad files needs anndata: pip install anndata") from e
    adata = anndata.read_h5ad(path)
    if label_column not in adata.obs.columns:
        raise ValueError(f"Column '{label_column}' not found in .obs.")
    X = adata.layers[layer] if layer else adata.X
    return X, adata.obs[label_column].astype(str).to_numpy(), list(adata.var_names)


def simulate_screen(
    n_treatments: int = 12,
    cells: int = 300,
    n_control: int = 2000,
    d: int = 30,
    intrinsic_dim: int = 5,
    seed: int = 0,
):
    """A simulated screen with known shifts, for demos and testing.

    Cells lie near a low-dimensional subspace of R^d, as expression data do.
    Treatment k moves the cells by a known distance along a random direction
    in that subspace, so the true W1 distance to the control equals the
    shift. Half of the treatments have no effect.

    Returns (X, labels, true_shift) where true_shift maps label -> distance.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, intrinsic_dim)))[0]

    def cells_at(n, offset):
        latent = rng.normal(size=(n, intrinsic_dim)) + offset
        return latent @ basis.T + 0.1 * rng.normal(size=(n, d))

    X = [cells_at(n_control, 0.0)]
    labels = ["control"] * n_control
    true_shift = {}
    shifts = np.r_[np.zeros(n_treatments // 2), np.linspace(0.3, 2.0, n_treatments - n_treatments // 2)]
    rng.shuffle(shifts)
    for k, s in enumerate(shifts):
        direction = rng.normal(size=intrinsic_dim)
        direction /= np.linalg.norm(direction)
        name = f"drug_{k + 1:02d}"
        X.append(cells_at(cells, s * direction))
        labels += [name] * cells
        true_shift[name] = float(s)
    return np.vstack(X), np.array(labels), true_shift
