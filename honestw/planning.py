"""Planning how many cells an experiment needs."""

from __future__ import annotations

import math

__all__ = ["hoeffding_width", "cells_needed"]


def hoeffding_width(B: float, n_test: int, m_test: int, alpha: float = 0.05) -> float:
    """Width subtracted in Theorem 4.3 when f takes values in [-B, B]."""
    return 2.0 * B * math.sqrt((1.0 / n_test + 1.0 / m_test) * math.log(1.0 / alpha) / 2.0)


def cells_needed(
    shift: float,
    B: float,
    alpha: float = 0.05,
    n_control: int | None = None,
    train_frac: float = 0.5,
    captured: float = 0.5,
) -> int | None:
    """Smallest number of treated cells for which a shift of size ``shift``
    is expected to be certified.

    The learned test function recovers only part of the true distance on
    held-out cells: in the simulations shipped with this package, between
    about 30% and 75% (median about 60%). ``captured`` is that fraction; the
    function returns the number of treated cells for which the Hoeffding
    width falls below ``captured * shift``. Use a smaller value to be
    cautious, or estimate it from a pilot experiment. B is
    the scale of the test function, which can be read off a pilot run
    (column "B" of a screen result) and is typically close to the spread of
    the data in the space used.

    Returns None if no number of treated cells is enough, which happens when
    the control group alone makes the width too large.
    """
    if shift <= 0 or B <= 0:
        raise ValueError("shift and B must be positive")
    if not 0 < captured <= 1:
        raise ValueError("captured must be in (0, 1]")
    test = 1.0 - train_frac
    target = (captured * shift / (2.0 * B)) ** 2 * 2.0 / math.log(1.0 / alpha)  # bound on 1/n + 1/m
    if n_control is not None:
        m_test = max(int(n_control * test), 1)
        target -= 1.0 / m_test
        if target <= 0:
            return None
        n_test = math.ceil(1.0 / target)
    else:
        # control group as large as the treated group
        n_test = math.ceil(2.0 / target)
    return math.ceil(n_test / test)
