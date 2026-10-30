"""Finite-sample confidence bounds for the Wasserstein-1 distance.

Implements the bounds of

    S. Kar, "Honest finite-sample inference for Wasserstein distances in
    general dimension", https://doi.org/10.2139/ssrn.7620318

Lower bound (Theorem 4.3, with the variance-adaptive variant of Remark 4.6a):
learn a 1-Lipschitz test function on a training half of the data, evaluate it
on the held-out half, and subtract a concentration term. The guarantee
P{W1(P, Q) >= L} >= 1 - alpha holds for every sample size, every dimension and
every pair of distributions. Nothing about the training step affects validity;
it only affects how tight the bound is.

Upper bound (Theorem 5.3, range version): requires that both distributions
live in a known set of diameter D.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import ot

__all__ = [
    "WitnessFunction",
    "LowerBoundResult",
    "UpperBoundResult",
    "fit_witness",
    "lower_bound_from_witness",
    "lower_bound_w1",
    "lower_bound_pvalue",
    "pvalue_from_witness",
    "resolve_method",
    "empirical_w1_upper",
    "upper_bound_w1",
]


# ---------------------------------------------------------------------------
# Test function learned on the training half
# ---------------------------------------------------------------------------


@dataclass
class WitnessFunction:
    """The 1-Lipschitz function f(z) = min_j {||z - y_j|| - v_j}  (eq. 2).

    It is centred at z0 and, when ``clip`` is True, clipped to [-B, B], which
    keeps it 1-Lipschitz and bounded. All of y, v, z0 and B are computed from
    the training half only.
    """

    y: np.ndarray  # training points of the second sample, shape (m1, d)
    v: np.ndarray  # dual variables for those points, shape (m1,)
    z0: np.ndarray  # centring point, shape (d,)
    f_z0: float  # value of the raw function at z0
    B: float  # clipping level (or half the known diameter when clip=False)
    clip: bool = True
    train_w1: float = float("nan")  # W1 between the two training halves
    var_x: float = float("nan")  # variance of f on the training X (used only to pick a method)
    var_y: float = float("nan")

    def _raw(self, Z: np.ndarray, chunk: int = 2048) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        out = np.empty(Z.shape[0])
        for s in range(0, Z.shape[0], chunk):
            dist = ot.dist(Z[s : s + chunk], self.y, metric="euclidean")
            out[s : s + chunk] = np.min(dist - self.v[None, :], axis=1)
        return out

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        vals = self._raw(Z) - self.f_z0
        if self.clip:
            vals = np.clip(vals, -self.B, self.B)
        return vals


def _subsample(X: np.ndarray, k: int | None, rng: np.random.Generator) -> np.ndarray:
    if k is None or X.shape[0] <= k:
        return X
    return X[rng.choice(X.shape[0], size=k, replace=False)]


def fit_witness(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    max_train: int | None = 3000,
    D: float | None = None,
    clip_quantile: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> WitnessFunction:
    """Learn the test function from the training halves.

    Solves the discrete optimal transport problem between the two training
    samples (Euclidean cost) and builds f from its dual solution. Training
    samples larger than ``max_train`` are subsampled to keep the transport
    solve fast; this only loosens the bound, it never breaks its validity.

    If ``D`` (a known bound on the diameter of the data range) is given, the
    function is not clipped and the width uses D (Theorem 4.3, last sentence).
    Otherwise B is set to the ``clip_quantile`` quantile of |f(z) - f(z0)| over
    the training points. Any B computed from the training half alone keeps the
    guarantee; a quantile below 1 stops a few outlying cells from inflating
    the width (0.95 worked best in simulations).
    """
    rng = np.random.default_rng(rng)
    X1 = _subsample(np.asarray(X_train, dtype=float), max_train, rng)
    Y1 = _subsample(np.asarray(Y_train, dtype=float), max_train, rng)
    if X1.shape[0] < 1 or Y1.shape[0] < 1:
        raise ValueError("Both training samples need at least one point.")

    a = np.full(X1.shape[0], 1.0 / X1.shape[0])
    b = np.full(Y1.shape[0], 1.0 / Y1.shape[0])
    M = ot.dist(X1, Y1, metric="euclidean")
    plan, log = ot.emd(a, b, M, log=True, numItermax=int(1e7))
    v = np.asarray(log["v"], dtype=float)
    train_w1 = float(np.sum(plan * M))

    z0 = np.vstack([X1, Y1]).mean(axis=0)
    w = WitnessFunction(y=Y1, v=v, z0=z0, f_z0=0.0, B=1.0, clip=False, train_w1=train_w1)
    w.f_z0 = float(w._raw(z0[None, :])[0])

    if D is not None:
        w.B = float(D) / 2.0
        w.clip = False
    else:
        centred = w._raw(np.vstack([X1, Y1])) - w.f_z0
        w.B = float(max(np.quantile(np.abs(centred), clip_quantile), 1e-12))
        w.clip = True
    w.var_x = float(np.var(w(X1), ddof=1)) if X1.shape[0] > 1 else w.B**2
    w.var_y = float(np.var(w(Y1), ddof=1)) if Y1.shape[0] > 1 else w.B**2
    return w


# ---------------------------------------------------------------------------
# Lower confidence bound
# ---------------------------------------------------------------------------


@dataclass
class LowerBoundResult:
    lower: float  # certified lower bound on W1(P, Q); max(lower, 0) is also valid
    statistic: float  # T1: mean of f on held-out X minus mean on held-out Y
    width: float  # amount subtracted from T1
    B: float
    n_test: int
    m_test: int
    alpha: float
    method: str
    train_w1: float

    def to_dict(self) -> dict:
        return asdict(self)


def _hoeffding_scale(B: float, n2: int, m2: int) -> float:
    """c such that the Hoeffding width is c * sqrt(log(1/alpha)).

    The values of f on the data lie in an interval of length 2B: [-B, B] when
    clipped, or an interval of length D = 2B when the data range is known.
    """
    return 2.0 * B * np.sqrt((1.0 / n2 + 1.0 / m2) / 2.0)


def _empirical_bernstein_dev(vals: np.ndarray, R: float, delta: float) -> float:
    """One-sided empirical Bernstein deviation (Maurer and Pontil, 2009, Thm 4).

    For i.i.d. values in an interval of length R, with probability >= 1 - delta
    the population mean is within this amount of the sample mean (one side).
    """
    n = vals.shape[0]
    if n < 2:
        return np.inf
    var = np.var(vals, ddof=1)
    log_term = np.log(2.0 / delta)
    return np.sqrt(2.0 * var * log_term / n) + 7.0 * R * log_term / (3.0 * (n - 1))


def resolve_method(witness: WitnessFunction, n_test: int, m_test: int, alpha: float, method: str) -> str:
    """Turn method="auto" into "hoeffding" or "bernstein".

    The choice uses the held-out sample sizes and the spread of f on the
    training cells only, never the held-out values, so it does not affect the
    guarantee. It picks the bound predicted to be narrower: Hoeffding for
    small groups, Bernstein once groups are large.
    """
    if method != "auto":
        return method
    if n_test < 2 or m_test < 2:
        return "hoeffding"
    hoef = _hoeffding_scale(witness.B, n_test, m_test) * np.sqrt(np.log(1.0 / alpha))
    R, log_term = 2.0 * witness.B, np.log(4.0 / alpha)

    def dev(var, n):
        return np.sqrt(2.0 * var * log_term / n) + 7.0 * R * log_term / (3.0 * (n - 1))

    bern = dev(witness.var_x, n_test) + dev(witness.var_y, m_test)
    return "bernstein" if bern < hoef else "hoeffding"


def _lower_from_values(fx: np.ndarray, fy: np.ndarray, B: float, alpha: float, method: str):
    """(lower, T1, width) from the values of f on the held-out samples."""
    n2, m2 = fx.shape[0], fy.shape[0]
    if n2 < 1 or m2 < 1:
        raise ValueError("Both held-out samples need at least one point.")
    T1 = float(fx.mean() - fy.mean())
    if method == "hoeffding":
        width = _hoeffding_scale(B, n2, m2) * np.sqrt(np.log(1.0 / alpha))
    elif method == "bernstein":
        R = 2.0 * B
        width = _empirical_bernstein_dev(fx, R, alpha / 2.0) + _empirical_bernstein_dev(fy, R, alpha / 2.0)
    else:
        raise ValueError("method must be 'hoeffding' or 'bernstein'")
    return T1 - float(width), T1, float(width)


def lower_bound_from_witness(
    witness: WitnessFunction,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alpha: float = 0.05,
    method: str = "auto",
) -> LowerBoundResult:
    """Evaluate a trained test function on held-out data (Theorem 4.3).

    method="hoeffding": the width of Theorem 4.3.
    method="bernstein": empirical Bernstein bound on each held-out mean at
    level alpha/2 (Remark 4.6a); tighter when f varies little on the data.
    method="auto": whichever of the two is predicted narrower from the
    training cells (see resolve_method).
    """
    fx = witness(X_test)
    fy = witness(Y_test)
    method = resolve_method(witness, fx.shape[0], fy.shape[0], alpha, method)
    lower, T1, width = _lower_from_values(fx, fy, witness.B, alpha, method)
    return LowerBoundResult(
        lower=lower,
        statistic=T1,
        width=width,
        B=witness.B,
        n_test=fx.shape[0],
        m_test=fy.shape[0],
        alpha=alpha,
        method=method,
        train_w1=witness.train_w1,
    )


def lower_bound_pvalue(statistic: float, B: float, n_test: int, m_test: int, delta: float = 0.0) -> float:
    """p-value for H0: W1(P, Q) <= delta, from the Hoeffding version.

    Obtained by inverting Theorem 4.3: the bound at level alpha exceeds delta
    exactly when alpha > p. Valid in finite samples under H0.
    """
    c = _hoeffding_scale(B, n_test, m_test)
    gap = statistic - delta
    if gap <= 0:
        return 1.0
    return float(np.exp(-((gap / c) ** 2)))


def pvalue_from_witness(
    witness: WitnessFunction,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    delta: float = 0.0,
    method: str = "auto",
    alpha: float = 0.05,
    tol: float = 1e-10,
) -> float:
    """p-value for H0: W1(P, Q) <= delta: the smallest alpha at which the
    lower bound exceeds delta. Closed form for Hoeffding, bisection otherwise
    (the bound increases with alpha, so the set of rejecting alphas is an
    interval). With method="auto", the method is fixed first using the
    nominal ``alpha`` and the training cells, then held for every alpha."""
    fx, fy = witness(X_test), witness(Y_test)
    method = resolve_method(witness, fx.shape[0], fy.shape[0], alpha, method)
    if method == "hoeffding":
        return lower_bound_pvalue(float(fx.mean() - fy.mean()), witness.B, fx.shape[0], fy.shape[0], delta)

    def rejects(a: float) -> bool:
        return _lower_from_values(fx, fy, witness.B, a, method)[0] > delta

    if not rejects(1.0 - 1e-12):
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mid <= 0:
            break
        if rejects(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def lower_bound_w1(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    train_frac: float = 0.5,
    method: str = "auto",
    max_train: int | None = 3000,
    D: float | None = None,
    clip_quantile: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> LowerBoundResult:
    """Certified lower confidence bound on W1(P, Q) from two samples.

    Splits each sample at random into a training part (fraction ``train_frac``)
    and a held-out part, learns the test function on the training part and
    evaluates it on the held-out part.

    Parameters
    ----------
    X, Y : arrays of shape (n, d) and (m, d)
        Samples from P and Q, e.g. treated and control cells in PCA space.
    alpha : float
        The bound holds with probability at least 1 - alpha.
    method : "auto" (default), "hoeffding" or "bernstein"
    max_train : int or None
        Cap on training points used in the transport solve.
    D : float or None
        Known diameter of the data range, if any.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xtr, Xte = _split(X, train_frac, rng)
    Ytr, Yte = _split(Y, train_frac, rng)
    w = fit_witness(Xtr, Ytr, max_train=max_train, D=D, clip_quantile=clip_quantile, rng=rng)
    return lower_bound_from_witness(w, Xte, Yte, alpha=alpha, method=method)


def _split(X: np.ndarray, train_frac: float, rng: np.random.Generator):
    n = X.shape[0]
    if n < 2:
        raise ValueError("Each sample needs at least two points to be split.")
    perm = rng.permutation(n)
    k = int(np.clip(round(train_frac * n), 1, n - 1))
    return X[perm[:k]], X[perm[k:]]


# ---------------------------------------------------------------------------
# Upper confidence bound (known data range)
# ---------------------------------------------------------------------------


def _mass_blocks(n: int, K: int, perm: np.ndarray):
    """Cut n points of mass 1/n into K consecutive pieces of mass 1/K each.

    Points that straddle a cut are split between two pieces. Returns, for each
    piece, (indices, masses) with masses summing to 1/K.
    """
    blocks = []
    edges = np.arange(n + 1) / n  # cumulative mass at point boundaries
    for k in range(K):
        lo, hi = k / K, (k + 1) / K
        start = np.searchsorted(edges, lo, side="right") - 1
        stop = np.searchsorted(edges, hi, side="left")
        idx = np.arange(max(start, 0), min(stop, n))
        mass = np.minimum(edges[idx + 1], hi) - np.maximum(edges[idx], lo)
        keep = mass > 1e-15
        blocks.append((perm[idx[keep]], mass[keep]))
    return blocks


def empirical_w1_upper(
    X: np.ndarray,
    Y: np.ndarray,
    max_pairs: int = 25_000_000,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, bool]:
    """Cost of a feasible coupling of the two empirical distributions.

    Exact W1(Pn, Qm) when n * m <= max_pairs. For larger samples, both samples
    are shuffled and cut into K pieces of equal mass, and the pieces are
    matched one to one; the result is the cost of a valid coupling, so it is
    at least W1(Pn, Qm), which is what Theorem 5.3 needs.

    Returns (value, exact).
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    if n * m <= max_pairs:
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        return float(ot.emd2(a, b, ot.dist(X, Y, metric="euclidean"), numItermax=int(1e7))), True

    K = int(np.ceil(np.sqrt(n * m / max_pairs)))
    bx = _mass_blocks(n, K, rng.permutation(n))
    by = _mass_blocks(m, K, rng.permutation(m))
    total = 0.0
    for (ix, mx), (iy, my) in zip(bx, by):
        Mk = ot.dist(X[ix], Y[iy], metric="euclidean")
        total += ot.emd2(mx * K, my * K, Mk, numItermax=int(1e7)) / K
    return float(total), False


@dataclass
class UpperBoundResult:
    upper: float
    plug_in: float  # cost of the feasible coupling used (exact W1(Pn, Qm) if exact)
    exact: bool
    width: float
    D: float
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def upper_bound_w1(
    X: np.ndarray,
    Y: np.ndarray,
    D: float,
    alpha: float = 0.05,
    max_pairs: int = 25_000_000,
    rng: np.random.Generator | int | None = None,
) -> UpperBoundResult:
    """Upper confidence bound on W1(P, Q) when both samples lie in a set of
    known diameter D (Theorem 5.3). Uses all points; no splitting.

    The guarantee depends on D being a true bound fixed before seeing the data,
    for example from an instrument's measurement range.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    W_hat, exact = empirical_w1_upper(X, Y, max_pairs=max_pairs, rng=rng)
    width = float(D) * np.sqrt((1.0 / n + 1.0 / m) * np.log(1.0 / alpha) / 2.0)
    return UpperBoundResult(upper=W_hat + width, plug_in=W_hat, exact=exact, width=width, D=float(D), alpha=alpha)
