"""Reproducible cohort, adjustment, and diagnostics for UCI diabetes encounters.

This dataset reports insulin dosage status over an entire encounter without
decision timestamps. Adjusted estimates are observational associations; causal
identification is not established. All modeled covariates are recorded before
or at admission according to the dataset description.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ("age_midpoint", "number_inpatient", "number_emergency", "number_outpatient")
CATEGORICAL = ("race", "gender", "admission_type_id", "admission_source_id")
EXCLUDED_DISCHARGES = {11, 13, 14, 19, 20, 21}


@dataclass(frozen=True)
class Settings:
    n_splits: int = 5
    seed: int = 42
    propensity_clip: float = .03
    matching_caliper_sd: float = .2


def prepare_cohort(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return analysis rows and a transparent row-count flow.

    Exclusions mirror the previous prediction project for comparability;
    discharge eligibility is post-treatment and can cause selection bias.
    """
    raw = pd.read_csv(path, na_values=["?"], low_memory=False)
    needed = {"patient_nbr", "age", "discharge_disposition_id", "insulin", "readmitted",
              *NUMERIC[1:], *CATEGORICAL}
    # age_midpoint is derived from age, rather than required in the input.
    needed.discard("age_midpoint")
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"Required CSV columns missing: {sorted(missing)}")

    flow = [{"stage": "Raw encounters", "encounters": len(raw),
             "patients": raw.patient_nbr.nunique()}]
    df = raw.loc[~raw.discharge_disposition_id.isin(EXCLUDED_DISCHARGES)].copy()
    flow.append({"stage": "Exclude death/hospice discharge codes",
                 "encounters": len(df), "patients": df.patient_nbr.nunique()})
    lower_age = pd.to_numeric(df.age.astype(str).str.extract(r"^\[(\d+)-")[0], errors="coerce")
    df = df.loc[lower_age >= 20].copy()
    df["age_midpoint"] = lower_age.loc[df.index] + 5
    flow.append({"stage": "Recorded age band starts at 20 or above",
                 "encounters": len(df), "patients": df.patient_nbr.nunique()})
    df = df.loc[df.insulin.isin(["Up", "Steady"])].copy()
    flow.append({"stage": "Insulin Up or Steady",
                 "encounters": len(df), "patients": df.patient_nbr.nunique()})
    df["treatment"] = (df.insulin == "Up").astype(int)
    df["readmitted_30d"] = (df.readmitted == "<30").astype(int)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(NUMERIC) + ["patient_nbr"]].isna().any().any():
        raise ValueError("Missing numeric covariates or patient IDs; review the cohort.")
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Missing").astype(str)
    if df.treatment.nunique() != 2:
        raise ValueError("Both insulin groups are required.")
    return df.reset_index(drop=True), pd.DataFrame(flow)


def _classifier() -> object:
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), list(NUMERIC)),
        ("category", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    return make_pipeline(preprocess, LogisticRegression(max_iter=1500))


def crossfit_nuisances(df: pd.DataFrame, settings: Settings) -> dict[str, np.ndarray]:
    """Predict propensity and two outcome risks on patients absent from training."""
    if not 0 < settings.propensity_clip < .5:
        raise ValueError("propensity_clip must be between 0 and 0.5.")
    t = df.treatment.to_numpy(dtype=int)
    y = df.readmitted_30d.to_numpy(dtype=int)
    groups = df.patient_nbr.to_numpy()
    features = df[list(NUMERIC) + list(CATEGORICAL)]
    p, m0, m1 = (np.full(len(df), np.nan) for _ in range(3))
    cv = StratifiedGroupKFold(n_splits=settings.n_splits, shuffle=True,
                             random_state=settings.seed)
    for train, test in cv.split(features, t, groups):
        assert not set(groups[train]).intersection(groups[test]), "Patient leakage between folds"
        ps_model = _classifier().fit(features.iloc[train], t[train])
        p[test] = ps_model.predict_proba(features.iloc[test])[:, 1]
        for group, target in ((0, m0), (1, m1)):
            subset = train[t[train] == group]
            if len(subset) == 0:
                raise ValueError("A training fold has no observations in a treatment group.")
            if np.unique(y[subset]).size == 1:
                target[test] = float(y[subset].mean())
            else:
                model = _classifier().fit(features.iloc[subset], y[subset])
                target[test] = model.predict_proba(features.iloc[test])[:, 1]
    if not all(np.isfinite(a).all() for a in (p, m0, m1)):
        raise RuntimeError("Cross-fitting did not return all predictions.")
    return {"propensity_raw": p, "propensity": np.clip(p, settings.propensity_clip,
                                                    1 - settings.propensity_clip),
            "risk0": m0, "risk1": m1}


def match_pairs(p: np.ndarray, t: np.ndarray, settings: Settings) -> np.ndarray:
    """Greedy nearest logit-score 1:1 matching, without control reuse."""
    if settings.matching_caliper_sd <= 0:
        raise ValueError("matching_caliper_sd must be positive.")
    logit = np.log(p / (1 - p))
    width = settings.matching_caliper_sd * np.std(logit, ddof=1)
    treated = np.flatnonzero(t == 1)
    controls = np.flatnonzero(t == 0)
    ordered = sorted((logit[j], int(j)) for j in controls)
    scores = [v for v, _ in ordered]
    indices = [j for _, j in ordered]
    pairs = []
    for i in np.random.default_rng(settings.seed).permutation(treated):
        if not scores:
            break
        pos = bisect_left(scores, logit[i])
        candidates = [k for k in (pos - 1, pos) if 0 <= k < len(scores)]
        k = min(candidates, key=lambda k: abs(scores[k] - logit[i]))
        if abs(scores[k] - logit[i]) <= width:
            pairs.append((int(i), indices.pop(k)))
            scores.pop(k)
    if not pairs:
        raise ValueError("No pairs matched within caliper; check overlap.")
    return np.asarray(pairs, dtype=int)


def _cluster_se(influence: np.ndarray, patient_id: np.ndarray) -> float:
    """Patient cluster robust SE for an average of row-level influence values."""
    codes, unique = pd.factorize(patient_id, sort=False)
    g = len(unique)
    if g < 2:
        raise ValueError("Cluster uncertainty requires at least two patients.")
    sums = np.bincount(codes, weights=influence)
    return float(np.sqrt(g / (g - 1) * np.sum(sums ** 2) / len(influence) ** 2))


def _ci(estimate: float, se: float) -> tuple[float, float]:
    return estimate - 1.96 * se, estimate + 1.96 * se


def _balance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Include all category indicators so imbalance is visible for every level."""
    return pd.get_dummies(df[list(NUMERIC) + list(CATEGORICAL)], columns=list(CATEGORICAL),
                          dtype=float)


def balance_table(df: pd.DataFrame, p: np.ndarray, pairs: np.ndarray) -> pd.DataFrame:
    x = _balance_matrix(df)
    t = df.treatment.to_numpy(dtype=int)
    n1, n0 = t.sum(), len(t) - t.sum()
    w = np.where(t == 1, (n1 / len(t)) / p, (n0 / len(t)) / (1 - p))
    matched = np.r_[pairs[:, 0], pairs[:, 1]]

    def smd(index: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        group = t[index]
        values = x.iloc[index].to_numpy(dtype=float)
        ww = np.ones(len(index)) if weights is None else weights
        means1 = np.average(values[group == 1], axis=0, weights=ww[group == 1])
        means0 = np.average(values[group == 0], axis=0, weights=ww[group == 0])
        base = x.to_numpy(dtype=float)
        denom = np.sqrt((base[t == 1].var(axis=0, ddof=1) +
                         base[t == 0].var(axis=0, ddof=1)) / 2)
        return np.divide(means1 - means0, denom, out=np.zeros_like(means1),
                         where=denom > 0)

    return pd.DataFrame({
        "covariate": x.columns,
        "smd_before": smd(np.arange(len(df))),
        "smd_after_iptw": smd(np.arange(len(df)), w),
        "smd_after_matching": smd(matched),
    })


def baseline_characteristics(df: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Numeric means and categorical proportions in the original two groups."""
    x = _balance_matrix(df)
    t = df.treatment.to_numpy(dtype=int)
    table = pd.DataFrame({
        "covariate": x.columns,
        "up_mean_or_share": x.loc[t == 1].mean().to_numpy(),
        "steady_mean_or_share": x.loc[t == 0].mean().to_numpy(),
        "unit": ["mean" if col in NUMERIC else "proportion" for col in x.columns],
    })
    return table.merge(balance, on="covariate", validate="one_to_one")


def analyze(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Compute crude, IPTW, AIPW risk differences and a matched ATT."""
    t = df.treatment.to_numpy(dtype=int)
    y = df.readmitted_30d.to_numpy(dtype=float)
    patient = df.patient_nbr.to_numpy()
    n, share = len(df), t.mean()
    nuisance = crossfit_nuisances(df, settings)
    raw_p, p, m0, m1 = (nuisance[k] for k in
                          ("propensity_raw", "propensity", "risk0", "risk1"))
    pairs = match_pairs(p, t, settings)

    naive1, naive0 = y[t == 1].mean(), y[t == 0].mean()
    naive = naive1 - naive0
    phi_naive = t * (y - naive1) / share - (1 - t) * (y - naive0) / (1 - share)

    w = np.where(t == 1, share / p, (1 - share) / (1 - p))
    d1, d0 = np.mean(t * w), np.mean((1 - t) * w)
    iptw1 = np.sum(t * w * y) / np.sum(t * w)
    iptw0 = np.sum((1 - t) * w * y) / np.sum((1 - t) * w)
    iptw = iptw1 - iptw0
    phi_iptw = t * w * (y - iptw1) / d1 - (1 - t) * w * (y - iptw0) / d0

    pseudo1 = m1 + t * (y - m1) / p
    pseudo0 = m0 + (1 - t) * (y - m0) / (1 - p)
    aipw1, aipw0 = pseudo1.mean(), pseudo0.mean()
    aipw = aipw1 - aipw0
    phi_aipw = pseudo1 - pseudo0 - aipw
    matched1, matched0 = y[pairs[:, 0]].mean(), y[pairs[:, 1]].mean()
    matched_effect = matched1 - matched0

    rows = []
    for method, target, value, risk1, risk0, influence in (
        ("Naive", "Crude risk difference", naive, naive1, naive0, phi_naive),
        ("IPTW", "ATE-like adjusted contrast", iptw, iptw1, iptw0, phi_iptw),
        ("AIPW", "ATE-like adjusted contrast", aipw, aipw1, aipw0, phi_aipw),
    ):
        se = _cluster_se(influence, patient)
        lo, hi = _ci(value, se)
        rows.append({"method": method, "target": target, "risk_difference": value,
                     "ci_low": lo, "ci_high": hi, "cluster_se": se,
                     "risk_up": risk1, "risk_steady": risk0})
    rows.append({"method": "PSM", "target": "Matched ATT-like contrast",
                 "risk_difference": matched_effect, "ci_low": np.nan, "ci_high": np.nan,
                 "cluster_se": np.nan, "risk_up": matched1, "risk_steady": matched0})
    effects = pd.DataFrame(rows)
    balance = balance_table(df, p, pairs)
    overlap_low = max(raw_p[t == 1].min(), raw_p[t == 0].min())
    overlap_high = min(raw_p[t == 1].max(), raw_p[t == 0].max())
    ess = {group: float(w[t == group].sum() ** 2 / np.sum(w[t == group] ** 2))
           for group in (0, 1)}
    diagnostics = {
        "encounters": int(n), "patients": int(df.patient_nbr.nunique()),
        "group_size": {"up": int(t.sum()), "steady": int((1 - t).sum())},
        "matched_pairs": len(pairs),
        "matched_fraction_of_up": len(pairs) / int(t.sum()),
        "propensity_min_max": {"up": [float(raw_p[t == 1].min()), float(raw_p[t == 1].max())],
                               "steady": [float(raw_p[t == 0].min()), float(raw_p[t == 0].max())]},
        "observed_overlap": [float(overlap_low), float(overlap_high)] if overlap_low <= overlap_high else None,
        "outside_observed_overlap_n": int(np.sum((raw_p < overlap_low) | (raw_p > overlap_high))),
        "extreme_propensity_n": int(np.sum((raw_p < .05) | (raw_p > .95))),
        "clipped_propensity_n": int(np.sum((raw_p < settings.propensity_clip) |
                                          (raw_p > 1 - settings.propensity_clip))),
        "max_stabilized_weight": float(w.max()),
        "p99_stabilized_weight": float(np.quantile(w, .99)),
        "effective_sample_size": {"up": ess[1], "steady": ess[0]},
        "max_abs_smd": {k: float(balance[k].abs().max()) for k in
                        ("smd_before", "smd_after_iptw", "smd_after_matching")},
        "covariates_over_smd_0_1": {k: int((balance[k].abs() >= .1).sum()) for k in
                                    ("smd_before", "smd_after_iptw", "smd_after_matching")},
        "ci_note": ("Naive/IPTW/AIPW Wald intervals use patient cluster influence approximations. "
                    "IPTW intervals omit nuisance model uncertainty; PSM inference is not reported."),
    }
    auxiliary = {"propensity_raw": raw_p, "weights": w, "treatment": t, "pairs": pairs}
    return effects, balance, diagnostics, auxiliary


def plot_results(effects: pd.DataFrame, balance: pd.DataFrame, aux: dict,
                 output_dir: Path) -> None:
    """Save four portfolio figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    p, t = aux["propensity_raw"], aux["treatment"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, label, color in ((0, "Steady", "#287c98"), (1, "Up", "#dc6a43")):
        ax.hist(p[t == group], bins=np.linspace(0, 1, 35), density=True,
                alpha=.52, label=label, color=color)
    ax.set(xlabel="Out-of-fold propensity score", ylabel="Density", title="Treatment-score overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "propensity_overlap.png")
    plt.close(fig)

    # Include strongly positive and negative pre-adjustment imbalances.
    show = balance.loc[balance.smd_before.abs().nlargest(min(20, len(balance))).index]
    show = show.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, len(show) * .3)))
    for key, label, color, marker in (
        ("smd_before", "Before", "#888888", "o"),
        ("smd_after_iptw", "IPTW", "#287c98", "s"),
        ("smd_after_matching", "PSM", "#dc6a43", "^"),
    ):
        ax.scatter(show[key].abs(), show.covariate, label=label, color=color,
                   marker=marker, s=28)
    ax.axvline(.1, linestyle="--", color="#555555", linewidth=1)
    ax.set(xlabel="Absolute standardized mean difference", title="Covariate balance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "covariate_balance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aux["weights"], bins=40, color="#287c98")
    ax.set(xlabel="Stabilized IPTW weight", ylabel="Encounter count", title="Weight distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_distribution.png")
    plt.close(fig)

    ordered = effects.set_index("method").loc[["Naive", "PSM", "IPTW", "AIPW"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, row) in enumerate(ordered.iterrows()):
        x = 100 * row.risk_difference
        if pd.notna(row.ci_low):
            ax.errorbar(x, i, xerr=[[100 * (row.risk_difference - row.ci_low)],
                                    [100 * (row.ci_high - row.risk_difference)]],
                        fmt="o", capsize=3, color="#287c98")
        else:
            ax.scatter(x, i, marker="o", facecolors="none", edgecolors="#dc6a43", s=60)
    ax.axvline(0, linewidth=1, color="#777777")
    ax.set(yticks=range(len(ordered)), yticklabels=ordered.index,
           xlabel="Risk difference, Up minus Steady (percentage points)",
           title="Observed and adjusted contrasts")
    ax.invert_yaxis()
    ax.text(.99, .02, "PSM: point estimate only", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "effect_comparison.png")
    plt.close(fig)
