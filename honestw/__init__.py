"""honestw: certified effect sizes for comparing groups of cells.

Finite-sample confidence bounds for the Wasserstein-1 distance, from
S. Kar, "Honest finite-sample inference for Wasserstein distances in general
dimension", https://doi.org/10.2139/ssrn.7620318
"""

from .bounds import (
    WitnessFunction,
    LowerBoundResult,
    UpperBoundResult,
    fit_witness,
    lower_bound_from_witness,
    lower_bound_w1,
    lower_bound_pvalue,
    pvalue_from_witness,
    empirical_w1_upper,
    upper_bound_w1,
)
from .screen import run_screen, benjamini_yekutieli
from .planning import cells_needed, hoeffding_width
from .data import load_csv, load_h5ad, simulate_screen

__version__ = "0.1.0"

__all__ = [
    "WitnessFunction",
    "LowerBoundResult",
    "UpperBoundResult",
    "fit_witness",
    "lower_bound_from_witness",
    "lower_bound_w1",
    "lower_bound_pvalue",
    "pvalue_from_witness",
    "empirical_w1_upper",
    "upper_bound_w1",
    "run_screen",
    "benjamini_yekutieli",
    "cells_needed",
    "hoeffding_width",
    "load_csv",
    "load_h5ad",
    "simulate_screen",
]
