# CARE-Lung

Reproducible code for **"CARE-Lung: Calibrated, Cross-Validated Cycle-to-Patient
Aggregation with Conformal Referral for Respiratory Sound Screening"**
(AII 2026, Springer CCIS).

CARE-Lung is a patient-level screening pipeline for the ICBHI 2017 respiratory
sound database: an isotonic-calibrated cycle classifier, a distributional
cycle-to-patient summary, a regularised patient-level aggregator, and a
split-conformal layer that can return `INCONCLUSIVE-REFER` instead of a forced
binary decision. Alongside the pipeline, the repository implements the
**device-confound audit** that bounds what the reported numbers mean.

## Headline results (pooled out-of-fold, all 126 patients)

| Metric | Value | 95% CI (patient-grouped bootstrap) |
|---|---|---|
| AUROC | 0.832 | 0.758 – 0.897 |
| PR-AUC | 0.958 | n/a |
| Sensitivity | 0.720 | 0.63 – 0.81 |
| Specificity | 0.769 | 0.60 – 0.92 |
| ECE | 0.165 | 0.115 – 0.233 |

**Read these with the controls.** Predicting every patient abnormal already
scores 0.794 accuracy / 0.885 F1, so accuracy and F1 are not meaningful gains on
this cohort. A baseline using **recording device alone and no audio** reaches
AUROC 0.784, and the paired difference against CARE-Lung is +0.047 with a 95%
interval of [-0.047, +0.141], which contains zero. Holding device constant
(Meditron, the only mixed-class cohort) gives AUROC 0.707 [0.576, 0.830].

What survives the device control is the methodology, not the headline: within a
single device, distributional aggregation still beats hard counting
(0.707 vs 0.579), calibration still improves 3.0× (ECE 0.478 → 0.157), and
conformal referral still cuts selective risk from 0.375 to 0.100.

## Data

The audio is the third-party [ICBHI 2017 Respiratory Sound
Database](https://bhichallenge.med.auth.gr/) and is **not** redistributed here.

`results/cycle_feature_cache.npz` contains the derived cycle-level feature
matrix (6,898 cycles × 31 features, with patient / recording / device / label
arrays). **Every analysis below runs from this cache**, so the audio is only
needed if you want to re-derive the features from scratch.

## Reproducing

```bash
pip install -r requirements.txt

# optional: re-derive the feature cache from raw ICBHI audio
python scripts/run_care_lung_study.py

# fixed-split + 5-fold patient-level cross-validation
python scripts/run_cv_study.py

# grouped bootstrap, paired tests, aggregator ablation, cross-conformal
python scripts/run_revision_analyses.py

# device-only baseline, paired test, device-controlled evaluation
python scripts/run_device_audit.py

# export the released audit artifacts (CSV)
python scripts/export_audit_artifacts.py

# regenerate every figure in the paper
python scripts/make_figures.py
```

All seeds are fixed (`SEED = 42`); results are deterministic.
`run_revision_analyses.py` reproduces the published headline metrics exactly.

## What each script does

| Script | Purpose |
|---|---|
| `run_care_lung_study.py` | Feature extraction from ICBHI audio, fixed-split training, cycle calibration |
| `run_cv_study.py` | Patient-level stratified 5-fold CV over all six aggregators |
| `run_revision_analyses.py` | Patient-grouped bootstrap CIs, paired ΔAUROC, within-fold manifest, pre-specified aggregator ablation, cross-conformal (marginal + Mondrian), risk-coverage |
| `run_device_audit.py` | Device-only baseline, paired test against it, device-controlled (within-Meditron) evaluation, per-device referral rates |
| `export_audit_artifacts.py` | Writes the released CSV artifacts |
| `make_figures.py` | Regenerates all nine paper figures as vector PDFs (≤12 cm wide, ≥7 pt text) |

## Released artifacts

`results/audit_artifacts/` is committed so the reported numbers can be checked
without re-running anything:

- `fold_split_manifest.csv`: patient IDs for every RF-train / calibration /
  threshold / test subset in every fold
- `patient_predictions.csv`: per-patient out-of-fold score and prediction for
  all six aggregators and six ablation variants, with fold index and device
- `fold_thresholds.csv`: the Youden thresholds selected within each fold
- `pooled_metrics_with_ci.csv`: pooled metrics with bootstrap intervals
- `risk_coverage_curve.csv`, `conformal_coverage.csv`: referral behaviour
- `device_confound_audit.csv`: the three device controls

## Limitations

126 patients (26 healthy); device is confounded with class. All 26 healthy
patients were recorded with a Meditron stethoscope, so three of the four device
cohorts are single-class. The pooled AUROC is not statistically separable from a
device-only baseline. Treat these results as evidence for an evaluation
methodology, not for a deployable system.

## Citation

```bibtex
@inproceedings{carelung2026,
  title     = {{CARE-Lung}: Calibrated, Cross-Validated Cycle-to-Patient
               Aggregation with Conformal Referral for Respiratory Sound
               Screening},
  booktitle = {Proc. 6th Int. Conf. on Applied Intelligence and Informatics
               (AII 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
