"""Run a simulated screen where the true shift of every treatment is known.

    python examples/simulated_screen.py

Compares the certified lower bound with the plug-in estimate and the truth.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import honestw as hw  # noqa: E402

X, labels, truth = hw.simulate_screen(n_treatments=12, cells=300, n_control=2000, d=30, seed=0)
res = hw.run_screen(X, labels, control="control", n_components=20, alpha=0.05, seed=0)
res["true_shift"] = res["treatment"].map(truth)

pd.set_option("display.width", 120)
print(res[["treatment", "true_shift", "certified_lower", "plug_in_w1", "q_value", "certified_shift"]].round(3))
print()
print("Certified lower bound above the true shift:", int((res["certified_lower"] > res["true_shift"]).sum()))
print("Plug-in estimate above the true shift:     ", int((res["plug_in_w1"] > res["true_shift"]).sum()))
print("Treatments with no effect called certified:", int(res.loc[res["true_shift"] == 0, "certified_shift"].sum()))
