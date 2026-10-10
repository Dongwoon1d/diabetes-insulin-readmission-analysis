"""Run the UCI diabetes observational comparison from the project root."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from src.analysis import (Settings, analyze, baseline_characteristics,
                          plot_results, prepare_cohort)


PROJECT_ROOT = Path(__file__).resolve().parent


def run(data_path: Path, output_root: Path, settings: Settings) -> dict:
    cohort, flow = prepare_cohort(data_path)
    effects, balance, diagnostics, aux = analyze(cohort, settings)
    result_dir, visual_dir = output_root / "results", output_root / "visuals"
    result_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    rates = (cohort.groupby("insulin")
             .agg(encounters=("readmitted_30d", "size"),
                  patients=("patient_nbr", "nunique"),
                  readmissions_30d=("readmitted_30d", "sum"),
                  readmission_rate=("readmitted_30d", "mean"))
             .reset_index())
    flow.to_csv(result_dir / "cohort_flow.csv", index=False)
    rates.to_csv(result_dir / "group_rates.csv", index=False)
    effects.to_csv(result_dir / "effect_estimates.csv", index=False)
    balance.to_csv(result_dir / "balance.csv", index=False)
    baseline_characteristics(cohort, balance).to_csv(
        result_dir / "baseline_characteristics.csv", index=False)
    with (result_dir / "diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2)
    plot_results(effects, balance, aux, visual_dir)

    # A small specification check documents the effect of the empirically chosen
    # propensity cap. It is not a sensitivity analysis for unmeasured confounding.
    sensitivity = []
    for clip in (.02, settings.propensity_clip, .05):
        if clip == settings.propensity_clip:
            trial_effects, trial_diag = effects, diagnostics
        else:
            trial_effects, _, trial_diag, _ = analyze(cohort, replace(settings, propensity_clip=clip))
        for _, estimate in trial_effects.iterrows():
            sensitivity.append({
                "propensity_clip": clip, "method": estimate.method,
                "risk_difference": estimate.risk_difference,
                "max_abs_smd_iptw": trial_diag["max_abs_smd"]["smd_after_iptw"],
                "max_stabilized_weight": trial_diag["max_stabilized_weight"],
            })
    pd.DataFrame(sensitivity).to_csv(result_dir / "clipping_sensitivity.csv", index=False)

    rows = []
    for _, r in effects.iterrows():
        interval = (f"[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]"
                    if pd.notna(r.ci_low) else "Not reported")
        rows.append(f"| {r.method} | {100*r.risk_difference:+.2f} | {interval} |")
    summary = (
        "# Insulin changes and 30-day readmission\n\n"
        "Companion observational analysis of the [diabetes readmission prediction "
        "project](https://github.com/Dongwoon1d/diabetes-readmission-prediction). "
        "Both projects use the same dataset and outcome; this one compares recorded "
        "insulin groups and does not establish a treatment effect.\n\n"
        f"Records: {len(cohort):,} encounters from {cohort.patient_nbr.nunique():,} patients.\n\n"
        "| Method | Up minus Steady (percentage points) | Approximate 95% CI (pp) |\n"
        "| --- | ---: | ---: |\n" + "\n".join(rows) + "\n\n"
        f"Matched pairs: {diagnostics['matched_pairs']:,}; "
        f"maximum absolute SMD after IPTW: {diagnostics['max_abs_smd']['smd_after_iptw']:.3f}; "
        f"after matching: {diagnostics['max_abs_smd']['smd_after_matching']:.3f}.\n\n"
        "The propensity cap of 0.03 was chosen after inspecting weight and balance "
        "diagnostics; 0.02 and 0.05 specifications are reported in "
        "`clipping_sensitivity.csv`. This does not address unmeasured confounding.\n\n"
        "The source records whether insulin dosage increased or stayed steady during a hospital "
        "encounter, but does not timestamp the dose decision or capture all pre-decision severity. "
        "These are observational contrasts under measured adjustment, not established treatment "
        "effects. The cohort also conditions on discharge status. PSM has a different matched "
        "population and no interval here; agreement among methods does not remove unmeasured "
        "confounding.\n\n"
        f"{diagnostics['ci_note']}\n"
    )
    (result_dir / "analysis_summary.md").write_text(summary)
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "diabetic_data.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    settings = Settings(n_splits=args.folds, seed=args.seed)
    report = run(args.data, args.output, settings)
    print("Completed:", report["encounters"], "encounters,", report["patients"], "patients")
    print("Matched pairs:", report["matched_pairs"])
    print("Results:", args.output / "results")
    print("Visuals:", args.output / "visuals")
