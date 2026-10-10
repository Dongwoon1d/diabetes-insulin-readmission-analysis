# Insulin changes and 30-day readmission

Companion observational analysis of the [diabetes readmission prediction project](https://github.com/Dongwoon1d/diabetes-readmission-prediction). Both projects use the same dataset and outcome; this one compares recorded insulin groups and does not establish a treatment effect.

Records: 40,472 encounters from 31,727 patients.

| Method | Up minus Steady (percentage points) | Approximate 95% CI (pp) |
| --- | ---: | ---: |
| Naive | +1.98 | [+1.15, +2.82] |
| IPTW | +1.11 | [+0.29, +1.92] |
| AIPW | +1.18 | [+0.38, +1.98] |
| PSM | +0.68 | Not reported |

Matched pairs: 10,702; maximum absolute SMD after IPTW: 0.082; after matching: 0.026.

The propensity cap of 0.03 was chosen after inspecting weight and balance diagnostics; 0.02 and 0.05 specifications are reported in `clipping_sensitivity.csv`. This does not address unmeasured confounding.

The source records whether insulin dosage increased or stayed steady during a hospital encounter, but does not timestamp the dose decision or capture all pre-decision severity. These are observational contrasts under measured adjustment, not established treatment effects. The cohort also conditions on discharge status. PSM has a different matched population and no interval here; agreement among methods does not remove unmeasured confounding.

Naive/IPTW/AIPW Wald intervals use patient cluster influence approximations. IPTW intervals omit nuisance model uncertainty; PSM inference is not reported.
