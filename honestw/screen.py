"""Run the certified comparison for every treatment in a screen.

Each treatment group is compared with one control group. Every cell is
assigned once, at random, to a training half or a held-out half:

* the PCA projection (if used) is fitted on training cells only;
* the test function for each treatment is learned on training cells only;
* the bound is computed on held-out cells only.

So the held-out cells never influence anything that the bound depends on, and
each lower bound keeps its finite-sample guarantee. All treatments share the
same control cells, so their tests are dependent; the Benjamini-Yekutieli
procedure controls the false discovery rate under any dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bounds import (
    fit_witness,
    lower_bound_from_witness,
    pvalue_from_witness,
    empirical_w1_upper,
    upper_bound_w1,
)
from .preprocess import split_mask, embed

__all__ = ["benjamini_yekutieli", "run_screen"]


def benjamini_yekutieli(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Yekutieli adjusted p-values and rejections at level q.

    Valid under arbitrary dependence between the tests.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p, p.astype(bool)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    ranked = p[order] * m * c_m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj, adj <= q


def run_screen(
    X,
    labels,
    control,
    alpha: float = 0.05,
    delta: float = 0.0,
    method: str = "auto",
    n_components: int | None = 20,
    train_frac: float = 0.5,
    max_train: int | None = 3000,
    fdr: float = 0.05,
    D: float | None = None,
    min_cells: int = 10,
    clip_quantile: float = 0.95,
    seed: int | None = 0,
    progress=None,
) -> pd.DataFrame:
    """Compare every treatment with the control group.

    Parameters
    ----------
    X : array or sparse matrix, shape (cells, features)
        Normalised measurements (e.g. log-normalised expression, or
        transformed cytometry intensities).
    labels : array of shape (cells,)
        Group of each cell.
    control : label of the control group.
    alpha : level of each lower (and upper) bound.
    delta : the shift that counts as meaningful; p-values test W1 <= delta.
    method : "auto" (picks the narrower bound using training cells only),
        "hoeffding" or "bernstein".
    n_components : number of principal components, or None to use X as is.
    D : known diameter of the data range in the space of X. Enables the upper
        bound. Only used when n_components is None, because a projection
        fitted to the data is not fixed in advance.
    min_cells : treatments with fewer cells are skipped.
    progress : optional callable(i, total, name) for progress reporting.

    Returns
    -------
    DataFrame with one row for each treatment, sorted by certified lower bound.
    """
    labels = np.asarray(labels).astype(str)
    control = str(control)
    rng = np.random.default_rng(seed)

    if np.sum(labels == control) < 2 * min_cells:
        raise ValueError(f"Control group '{control}' has too few cells.")

    train = split_mask(labels, train_frac, rng)
    Z = embed(X, train, n_components)

    use_upper = D is not None and n_components is None

    ctrl = labels == control
    C_tr, C_te, C_all = Z[ctrl & train], Z[ctrl & ~train], Z[ctrl]

    groups = [g for g in pd.unique(labels) if g != control]
    rows = []
    for i, g in enumerate(groups):
        if progress is not None:
            progress(i, len(groups), g)
        sel = labels == g
        n = int(sel.sum())
        if n < min_cells:
            continue
        T_tr, T_te = Z[sel & train], Z[sel & ~train]
        if T_tr.shape[0] < 1 or T_te.shape[0] < 2:
            continue
        w = fit_witness(
            T_tr, C_tr, max_train=max_train, D=D if use_upper else None, clip_quantile=clip_quantile, rng=rng
        )
        lb = lower_bound_from_witness(w, T_te, C_te, alpha=alpha, method=method)
        pval = pvalue_from_witness(w, T_te, C_te, delta=delta, method=method, alpha=alpha)

        row = {
            "treatment": g,
            "n_cells": n,
            "certified_lower": max(lb.lower, 0.0),
            "raw_lower": lb.lower,
            "heldout_statistic": lb.statistic,
            "width": lb.width,
            "B": lb.B,
            "bound_used": lb.method,
            "p_value": pval,
        }
        if use_upper:
            ub = upper_bound_w1(Z[sel], C_all, D=D, alpha=alpha, rng=rng)
            row["plug_in_w1"] = ub.plug_in
            row["upper"] = ub.upper
        else:
            row["plug_in_w1"], _ = empirical_w1_upper(Z[sel], C_all, rng=rng)
        rows.append(row)

    if progress is not None:
        progress(len(groups), len(groups), "done")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"], df["certified_shift"] = benjamini_yekutieli(df["p_value"].to_numpy(), fdr)
    df = df.sort_values(["certified_lower", "heldout_statistic"], ascending=False).reset_index(drop=True)
    df.attrs.update(
        alpha=alpha,
        delta=delta,
        method=method,
        n_components=n_components,
        fdr=fdr,
        n_control=int(ctrl.sum()),
        upper_bound=use_upper,
    )
    return df
