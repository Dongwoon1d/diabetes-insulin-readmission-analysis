# GitHub and portfolio presentation

This repository is a **companion analysis** of the [existing readmission prediction project](https://github.com/Dongwoon1d/diabetes-readmission-prediction). Both use the same dataset and outcome. Present them as two related questions, not as two independent clinical datasets or causal discoveries.

## New repository settings

**Suggested repository name:** `diabetes-insulin-readmission-analysis`

**GitHub description (paste into the About field):**

> Observational follow-up to my diabetes readmission prediction project: insulin dose changes, adjusted readmission comparisons, and study limitations.

**Suggested topics:** `healthcare-data`, `observational-study`, `propensity-score`, `readmission`, `python`

**Optional website field:** Leave blank unless you have a portfolio page for this project. The link to the prediction project already appears at the start of this README.

**Publish order:** Create an empty public GitHub repository with the suggested name, upload the contents of this project's folder (so `README.md` is at the repository root), and then add the reciprocal link to the existing project's README. If GitHub offers to create a README during repository creation, leave that option unchecked because this project already includes one.

## Add this to the existing prediction project's README

Place this short section after its project overview. Once the new repository is published at the suggested URL, paste the following into the **existing** repository's `README.md`:

```markdown
## Companion analysis

I also explored a different question using the same dataset and 30-day readmission outcome: how do readmission rates compare between encounters with an insulin dose increase and those with no insulin dose change? The [insulin and readmission observational analysis](https://github.com/Dongwoon1d/diabetes-insulin-readmission-analysis) compares the groups after measured-variable adjustment and explains why the data do not establish a treatment effect. This prediction model and its SHAP explanations do not estimate the effect of changing insulin.
```

If you choose a different repository name, update the link before pasting. This file is a copy-ready guide; it does not change the existing GitHub repository.

## Portfolio card

**Title:** Insulin Changes and 30-Day Readmission

**One-line description:** Companion to my readmission prediction model, this observational analysis compares insulin `Up` and `Steady` records using matching, weighting, and diagnostics while evaluating limits to causal interpretation.

**Place beside:** Diabetes 30-Day Readmission Prediction. Label them as a connected two-part healthcare analysis: (1) predicting risk, (2) examining a treatment-related association. Do not count the shared dataset twice when describing the breadth of your portfolio.
