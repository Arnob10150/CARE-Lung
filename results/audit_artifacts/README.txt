CARE-Lung audit artifacts
=========================

fold_split_manifest.csv   Patient identifiers for every within-fold subset
                          (rf_train / calibration / threshold / test), with
                          label and cycle count. Table 2 in the paper is a
                          summary of this file.
patient_predictions.csv   One row per patient: pooled out-of-fold score and
                          hard prediction for all six aggregators and all six
                          ablation variants, plus fold index and device.
fold_thresholds.csv       Youden thresholds selected within each fold.
pooled_metrics_with_ci.csv  Tables 3-5: pooled metrics with patient-grouped
                          bootstrap 95% intervals (4,000 replicates).
risk_coverage_curve.csv   Figure 4(a): selective risk vs. coverage.
conformal_coverage.csv    Table 5 / Figure 2(b): marginal and Mondrian
                          class-conditional coverage, referral, selective risk.
device_confound_audit.csv Section 4.7: trivial and device-only baselines, the
                          paired test against device-only, device-controlled
                          (within-Meditron) metrics, and per-device referral.

Regenerate with:
  python scripts/run_revision_analyses.py
  python scripts/run_device_audit.py
  python scripts/export_audit_artifacts.py
All seeds fixed (SEED=42); results are deterministic.
