"""Web app: certified effect sizes for drug and gene perturbation screens.

Run locally with:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import honestw as hw  # noqa: E402

st.set_page_config(page_title="Honest Wasserstein", layout="wide")

st.title("Honest Wasserstein")
st.markdown(
    "Did a treatment really change the cells, and by **at least** how much? "
    "Upload control and treated cells and get, for each treatment, a lower bound on the "
    "Wasserstein distance to the control that holds with the stated confidence for any "
    "sample size and any distribution, with no modelling assumptions."
)

tab_run, tab_plan, tab_about = st.tabs(["Analyse a screen", "Plan an experiment", "How it works"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _simulated(seed: int):
    return hw.simulate_screen(seed=seed)


@st.cache_data(show_spinner=False)
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def _read_h5ad(file_bytes: bytes):
    import anndata

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as fh:
        fh.write(file_bytes)
        path = fh.name
    try:
        return anndata.read_h5ad(path)
    finally:
        os.unlink(path)


with tab_run:
    source = st.radio(
        "Data",
        ["Simulated example screen", "Upload CSV (one row for each cell)", "Upload AnnData (.h5ad)"],
        horizontal=True,
    )

    X = labels = None
    truth = None
    feature_names: list[str] = []

    if source.startswith("Simulated"):
        X, labels, truth = _simulated(0)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        st.caption(
            "12 simulated drugs, 300 cells each, against 2,000 control cells in 30 dimensions. "
            "Half the drugs have no effect; the true shift of each drug is known, so you can "
            "check that the certified bound never exceeds it."
        )
        control = "control"
    elif source.startswith("Upload CSV"):
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            df_in = _read_csv(up.getvalue())
            label_col = st.selectbox("Column with the group of each cell", df_in.columns)
            numeric = [c for c in df_in.select_dtypes("number").columns if c != label_col]
            feature_names = st.multiselect("Measurement columns", numeric, default=numeric)
            if feature_names:
                X = df_in[feature_names].to_numpy(dtype=float)
                labels = df_in[label_col].astype(str).to_numpy()
    else:
        up = st.file_uploader("AnnData file (normalised expression in .X)", type=["h5ad"])
        if up is not None:
            adata = _read_h5ad(up.getvalue())
            label_col = st.selectbox("Column of .obs with the group of each cell", list(adata.obs.columns))
            layer = st.selectbox("Matrix", ["X"] + list(adata.layers.keys()))
            X = adata.X if layer == "X" else adata.layers[layer]
            labels = adata.obs[label_col].astype(str).to_numpy()
            feature_names = list(adata.var_names)

    if X is not None:
        groups = pd.Series(labels).value_counts()
        if not source.startswith("Simulated"):
            control = st.selectbox("Control group", groups.index.tolist())
        st.write(
            f"{X.shape[0]:,} cells, {X.shape[1]:,} features, {len(groups) - 1} treatments, "
            f"{int(groups.get(control, 0)):,} control cells."
        )

        with st.expander("Settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            confidence = c1.select_slider("Confidence", [0.8, 0.9, 0.95, 0.99], value=0.95)
            fdr = c1.select_slider("False discovery rate across the screen", [0.01, 0.05, 0.1, 0.2], value=0.05)
            use_pca = c2.checkbox("Project onto principal components", value=X.shape[1] > 50)
            n_comp = c2.number_input("Number of components", 2, 100, 20, disabled=not use_pca)
            delta = c2.number_input(
                "Smallest shift that matters (0 = any shift)", min_value=0.0, value=0.0, step=0.1
            )
            method = c3.radio(
                "Bound",
                ["auto", "hoeffding", "bernstein"],
                help="Auto picks whichever bound is predicted narrower, using the training cells only, "
                "so the guarantee is kept. Hoeffding suits small groups; Bernstein suits large ones.",
            )
            seed = c3.number_input("Random seed", 0, 10_000, 0)
            D = None
            if not use_pca:
                known = c3.checkbox(
                    "Values have a known range (enables upper bounds)",
                    help="For example the measurement range of a cytometer after transformation. "
                    "It must be fixed before looking at the data.",
                )
                if known:
                    lo = c3.number_input("Lowest possible value", value=0.0)
                    hi = c3.number_input("Highest possible value", value=1.0)
                    if hi > lo:
                        D = (hi - lo) * math.sqrt(X.shape[1])
                        c3.caption(f"Diameter of the data range: {D:.3g}")

        if st.button("Run", type="primary"):
            bar = st.progress(0.0, text="Starting")

            def progress(i, total, name):
                bar.progress(min(i / max(total, 1), 1.0), text=f"Treatment {i} of {total}: {name}")

            try:
                res = hw.run_screen(
                    X,
                    labels,
                    control,
                    alpha=1 - confidence,
                    delta=delta,
                    method=method,
                    n_components=int(n_comp) if use_pca else None,
                    fdr=fdr,
                    D=D,
                    seed=int(seed),
                    progress=progress,
                )
            except ValueError as err:
                st.error(str(err))
                res = None
            bar.empty()
            st.session_state["result"] = res
            st.session_state["truth"] = truth
            st.session_state["settings"] = dict(confidence=confidence, fdr=fdr, delta=delta)

    res = st.session_state.get("result")
    if res is not None and not res.empty:
        s = st.session_state["settings"]
        truth = st.session_state.get("truth")
        n_cert = int(res["certified_shift"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Treatments analysed", len(res))
        m2.metric(f"Certified shift > {s['delta']:g}", n_cert)
        m3.metric("Lower bound above 0", int((res["raw_lower"] > 0).sum()))

        st.markdown(
            f"**Reading the results.** For each treatment, *certified lower bound* is a value the true "
            f"distance to the control exceeds with {s['confidence']:.0%} confidence. *Certified shift* "
            f"marks treatments whose shift exceeds {s['delta']:g} while keeping the expected share of "
            f"false calls among them below {s['fdr']:.0%}. The *plug-in estimate* is the usual "
            f"distance between the samples; it is biased upward, so it is shown for comparison only."
        )

        plot = res.copy()
        if truth is not None:
            plot["true_shift"] = plot["treatment"].map(truth)
        order = plot["treatment"].tolist()
        base = alt.Chart(plot).encode(
            y=alt.Y("treatment:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=200))
        )
        layers = [
            base.mark_bar(color="#2a7f62").encode(
                x=alt.X("certified_lower:Q", title="Distance to control"),
                tooltip=["treatment", "certified_lower", "plug_in_w1", "n_cells", "q_value"],
            ),
            base.mark_tick(color="#999999", thickness=3, size=18).encode(x="plug_in_w1:Q"),
        ]
        if "upper" in plot:
            layers.append(base.mark_tick(color="#c0392b", thickness=3, size=18).encode(x="upper:Q"))
        if truth is not None:
            layers.append(base.mark_point(color="#1f4e9c", filled=True, size=70).encode(x="true_shift:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=max(220, 30 * len(plot))), use_container_width=True)
        legend = "Green bar: certified lower bound. Grey tick: plug-in estimate."
        if "upper" in plot:
            legend += " Red tick: upper bound."
        if truth is not None:
            legend += " Blue dot: true shift (simulation only)."
        st.caption(legend)

        cols = ["treatment", "n_cells", "certified_lower", "plug_in_w1"]
        cols += ["upper"] if "upper" in res else []
        cols += ["p_value", "q_value", "certified_shift", "width", "B", "bound_used"]
        st.dataframe(res[cols].round(4), use_container_width=True, hide_index=True)
        st.download_button(
            "Download results (CSV)", res.to_csv(index=False).encode(), "honest_wasserstein_results.csv", "text/csv"
        )
        st.session_state["median_B"] = float(res["B"].median())
    elif res is not None:
        st.warning("No treatment had enough cells to analyse.")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

with tab_plan:
    st.markdown(
        "How many treated cells are needed before a shift of a given size can be certified? "
        "The answer depends on the scale **B** of the data in the space you analyse. Run a pilot "
        "or the example first and B is filled in from its results."
    )
    p1, p2 = st.columns(2)
    shift = p1.number_input("Shift you want to detect", min_value=0.01, value=0.5, step=0.1)
    B = p1.number_input("Scale B", min_value=0.01, value=float(round(st.session_state.get("median_B", 2.0), 3)))
    conf = p2.select_slider("Confidence ", [0.8, 0.9, 0.95, 0.99], value=0.95)
    n_ctrl = p2.number_input("Control cells available", min_value=10, value=2000, step=100)
    captured = p1.slider(
        "Share of the shift the method recovers",
        0.2,
        1.0,
        0.5,
        0.05,
        help="On held-out cells the learned function recovers only part of the true shift: "
        "about 30-75% in the simulations (median about 60%). Lower values plan more cautiously.",
    )
    need = hw.cells_needed(shift, B, alpha=1 - conf, n_control=int(n_ctrl), captured=captured)
    if need is None:
        st.error("Even with unlimited treated cells this is not enough: add control cells.")
    else:
        st.success(f"At least **{need:,} treated cells**.")
    st.caption(
        "This is an estimate, not a guarantee: whether a shift is certified depends on the data. "
        "Treat it as a minimum and plan some margin."
    )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown(
        """
**The problem.** With thousands of cells, standard tests call almost every treatment
significant, even when the change is tiny. The usual Wasserstein distance between
samples is biased upward in high dimension, and bootstrap intervals around it can miss
the true value entirely.

**The method** (Kar, *Honest finite-sample inference for Wasserstein distances in general
dimension*, [doi:10.2139/ssrn.7620318](https://doi.org/10.2139/ssrn.7620318)):

1. Every cell is put at random into a training half or a held-out half.
2. If principal components are used, they are fitted on the training half only.
3. For each treatment, an optimal transport problem between training cells gives a
   1-Lipschitz function that separates treated from control cells.
4. On held-out cells, the difference of its averages, minus a concentration term, is a
   lower bound on the Wasserstein-1 distance. It holds with the stated confidence for every
   sample size, every dimension and every pair of distributions (Theorem 4.3).
5. Across the screen, the Benjamini-Yekutieli procedure controls false discoveries, which
   stays valid even though all treatments share the same control cells.

**Limits.** Splitting the cells costs power, so weak effects in small groups are not
certified. The distance is measured in the space you analyse (for example principal
components), not gene by gene. Upper bounds need a value range known in advance.
"""
    )
