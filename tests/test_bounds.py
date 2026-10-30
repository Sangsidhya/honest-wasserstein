import numpy as np
import pytest

import honestw as hw
from honestw.bounds import _mass_blocks, fit_witness


def _translation_pair(n, d, shift, rng):
    """P = Unif[0,1]^d, Q = P translated by `shift` along the first axis: W1 = shift."""
    X = rng.uniform(size=(n, d))
    Y = rng.uniform(size=(n, d))
    Y[:, 0] += shift
    return X, Y


@pytest.mark.parametrize("method", ["hoeffding", "bernstein", "auto"])
def test_lower_bound_coverage(method):
    """The lower bound should fall below the true distance in at least 95% of runs."""
    rng = np.random.default_rng(0)
    true_w1, misses, reps = 0.5, 0, 100
    for _ in range(reps):
        X, Y = _translation_pair(300, 5, true_w1, rng)
        if hw.lower_bound_w1(X, Y, alpha=0.05, method=method, rng=rng).lower > true_w1:
            misses += 1
    assert misses / reps <= 0.05 + 0.05  # allow Monte Carlo slack


def test_lower_bound_detects_large_shift():
    rng = np.random.default_rng(1)
    X, Y = _translation_pair(2000, 3, 1.0, rng)
    res = hw.lower_bound_w1(X, Y, rng=rng)
    assert 0 < res.lower <= 1.0


def test_null_rarely_positive():
    rng = np.random.default_rng(2)
    positives = 0
    for _ in range(50):
        X, Y = _translation_pair(300, 5, 0.0, rng)
        positives += hw.lower_bound_w1(X, Y, rng=rng).lower > 0
    assert positives <= 5


def test_witness_is_lipschitz_and_bounded():
    rng = np.random.default_rng(3)
    X, Y = _translation_pair(200, 4, 0.3, rng)
    w = fit_witness(X, Y, rng=rng)
    Z1, Z2 = rng.normal(size=(500, 4)), rng.normal(size=(500, 4))
    lhs = np.abs(w(Z1) - w(Z2))
    rhs = np.linalg.norm(Z1 - Z2, axis=1)
    assert np.all(lhs <= rhs + 1e-9)
    assert np.all(np.abs(w(Z1)) <= w.B + 1e-12)


def test_witness_reaches_training_distance():
    """Unclipped, the function's mean difference on training data equals W1 between them."""
    rng = np.random.default_rng(4)
    X, Y = _translation_pair(150, 3, 0.4, rng)
    w = fit_witness(X, Y, max_train=None, D=10.0, rng=rng)
    assert w(X).mean() - w(Y).mean() == pytest.approx(w.train_w1, abs=1e-6)


def test_pvalue_matches_bound():
    rng = np.random.default_rng(5)
    X, Y = _translation_pair(400, 3, 0.6, rng)
    Xtr, Xte, Ytr, Yte = X[:200], X[200:], Y[:200], Y[200:]
    w = fit_witness(Xtr, Ytr, rng=rng)
    for method in ["hoeffding", "bernstein"]:
        p = hw.pvalue_from_witness(w, Xte, Yte, delta=0.1, method=method)
        if 0 < p < 0.999:
            above = hw.lower_bound_from_witness(w, Xte, Yte, alpha=min(p * 1.01, 0.999), method=method).lower
            below = hw.lower_bound_from_witness(w, Xte, Yte, alpha=p * 0.99, method=method).lower
            assert above > 0.1 >= below


def test_mass_blocks_are_a_partition():
    n, K = 37, 5
    perm = np.random.default_rng(0).permutation(n)
    blocks = _mass_blocks(n, K, perm)
    total = np.zeros(n)
    for idx, mass in blocks:
        assert mass.sum() == pytest.approx(1 / K)
        np.add.at(total, idx, mass)
    assert np.allclose(total, 1 / n)


def test_blocked_coupling_is_above_exact():
    rng = np.random.default_rng(6)
    X, Y = rng.normal(size=(300, 3)), rng.normal(size=(250, 3)) + 0.5
    exact, is_exact = hw.empirical_w1_upper(X, Y)
    blocked, is_exact2 = hw.empirical_w1_upper(X, Y, max_pairs=10_000, rng=rng)
    assert is_exact and not is_exact2
    assert blocked >= exact - 1e-9


def test_upper_bound_coverage():
    rng = np.random.default_rng(7)
    d, true_w1 = 2, 0.3
    D = np.sqrt((1 + true_w1) ** 2 + (d - 1))  # diameter of the box holding both samples
    misses = 0
    for _ in range(50):
        X, Y = _translation_pair(300, d, true_w1, rng)
        misses += hw.upper_bound_w1(X, Y, D=D, rng=rng).upper < true_w1
    assert misses <= 5


def test_benjamini_yekutieli():
    p = np.array([0.0001, 0.2, 0.001, 0.9])
    adj, rej = hw.benjamini_yekutieli(p, 0.05)
    assert rej.tolist() == [True, False, True, False]
    assert np.all(adj >= p)


def test_screen_runs_and_never_overstates():
    X, labels, truth = hw.simulate_screen(n_treatments=6, cells=200, n_control=800, seed=0)
    res = hw.run_screen(X, labels, "control", n_components=10, seed=0)
    assert len(res) == 6
    for _, row in res.iterrows():
        assert row["certified_lower"] <= truth[row["treatment"]] + 0.05
    assert res.loc[res["treatment"].map(truth) == 0, "certified_shift"].sum() == 0


def test_cells_needed():
    n = hw.cells_needed(0.5, B=1.0, alpha=0.05, n_control=5000, captured=1.0)
    assert n is not None and n > 0
    w = hw.hoeffding_width(1.0, int(n * 0.5), int(5000 * 0.5), 0.05)
    assert w <= 0.5 + 1e-6
    assert hw.cells_needed(0.5, B=1.0, n_control=5000, captured=0.5) > n
    assert hw.cells_needed(0.01, B=1.0, n_control=20) is None
