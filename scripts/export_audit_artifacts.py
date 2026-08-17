"""Export the audit artifacts promised in the paper's 'Released Artifacts'
section: per-fold split manifests with patient identifiers, per-fold decision
thresholds, per-patient out-of-fold predictions, and the pooled metric table
with patient-grouped bootstrap intervals.

Split reconstruction uses the same seeds and the same call sequence as
run_cv_study.py / run_revision_analyses.py, so the identifiers written here are
exactly the cohorts used to produce the reported numbers.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "audit_artifacts"
SEED = 42
N_FOLDS = 5


def main():
    OUT.mkdir(exist_ok=True)

    cache = np.load(RESULTS / "cycle_feature_cache.npz", allow_pickle=True)
    patients = cache["patient"]
    y_patient_cycle = cache["y_patient"].astype(int)
    patient_ids_all = np.array(sorted(set(patients.tolist())))
    patient_labels = np.array([int(y_patient_cycle[patients == p][0]) for p in patient_ids_all])

    # ---- 1. split manifest with patient identifiers ------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (trainval_pos, test_pos) in enumerate(skf.split(patient_ids_all, patient_labels)):
        test_patients = patient_ids_all[test_pos]
        trainval_patients = patient_ids_all[trainval_pos]
        trainval_labels = patient_labels[trainval_pos]
        rf_train_p, rest_p, _, rest_y = train_test_split(
            trainval_patients, trainval_labels, test_size=0.40,
            random_state=SEED + fold, stratify=trainval_labels)
        cal_p, thr_p, _, _ = train_test_split(
            rest_p, rest_y, test_size=0.5,
            random_state=SEED + 100 + fold, stratify=rest_y)
        for role, group in (("rf_train", rf_train_p), ("calibration", cal_p),
                            ("threshold", thr_p), ("test", test_patients)):
            for p in sorted(group.tolist()):
                i = int(np.where(patient_ids_all == p)[0][0])
                rows.append({"fold": fold, "subset": role, "patient_id": p,
                             "label": int(patient_labels[i]),
                             "n_cycles": int(np.sum(patients == p))})
    with (OUT / "fold_split_manifest.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "subset", "patient_id", "label", "n_cycles"])
        w.writeheader()
        w.writerows(rows)
    print(f"fold_split_manifest.csv: {len(rows)} rows")

    # ---- 2. per-patient out-of-fold predictions ----------------------------
    oof = np.load(RESULTS / "revision_oof.npz", allow_pickle=True)
    names = [str(n) for n in oof["model_names"]]
    header = ["patient_id", "y_true", "fold", "device"]
    for n in names:
        header += [f"prob::{n}", f"pred::{n}"]
    with (OUT / "patient_predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, pid in enumerate(oof["patient_ids"]):
            row = [pid, int(oof["y_true"][i]), int(oof["fold"][i]), str(oof["device"][i])]
            for j in range(len(names)):
                row += [round(float(oof[f"prob__{j}"][i]), 6), int(oof[f"pred__{j}"][i])]
            w.writerow(row)
    print(f"patient_predictions.csv: {len(oof['patient_ids'])} patients x {len(names)} models")

    # ---- 3. thresholds and 4. pooled metrics with intervals ----------------
    rev = json.loads((RESULTS / "revision_analyses.json").read_text())
    with (OUT / "fold_thresholds.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"fold{k}" for k in range(N_FOLDS)])
        for n, v in rev["models"].items():
            w.writerow([n] + v["fold_thresholds"])

    keys = ["Accuracy", "Precision", "Sensitivity", "F1", "Specificity", "AUROC", "PR_AUC", "ECE"]
    with (OUT / "pooled_metrics_with_ci.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + sum([[k, f"{k}_lo95", f"{k}_hi95"] for k in keys], []))
        for n, v in rev["models"].items():
            row = [n]
            for k in keys:
                row += [v["pooled"][k], v["bootstrap_ci95"][k]["lo"], v["bootstrap_ci95"][k]["hi"]]
            w.writerow(row)

    # ---- 5. conformal + risk-coverage curves -------------------------------
    with (OUT / "risk_coverage_curve.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["band", "coverage", "risk", "sens", "spec"])
        w.writeheader()
        w.writerows(rev["risk_coverage"])

    with (OUT / "conformal_coverage.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "alpha", "cov_abnormal", "cov_healthy", "refer_rate", "selective_risk"])
        for a, v in rev["cross_conformal_marginal"].items():
            w.writerow(["marginal", a, round(v["cov_abn"]["mean"], 4), round(v["cov_hea"]["mean"], 4),
                        round(v["refer"]["mean"], 4), round(v["sel_risk"]["mean"], 4)])
        for a, v in rev["cross_conformal_mondrian"].items():
            w.writerow(["mondrian", a, round(v["cov_abn"]["mean"], 4), round(v["cov_hea"]["mean"], 4),
                        round(v["refer"]["mean"], 4), round(v["sel_risk"]["mean"], 4)])

    # ---- 6. device-confound audit ------------------------------------------
    dev = json.loads((RESULTS / "device_audit.json").read_text())
    with (OUT / "device_confound_audit.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["control", "quantity", "value", "lo95", "hi95"])
        t = dev["trivial_all_abnormal"]
        for k in ["accuracy", "f1", "auroc", "pr_auc"]:
            w.writerow(["trivial_all_abnormal", k, round(t[k], 4), "", ""])
        d0 = dev["device_only_baseline"]
        w.writerow(["device_only", "auroc", round(d0["auroc"], 4),
                    round(d0["auroc_ci"]["lo"], 4), round(d0["auroc_ci"]["hi"], 4)])
        w.writerow(["device_only", "pr_auc", round(d0["pr_auc"], 4),
                    round(d0["pr_auc_ci"]["lo"], 4), round(d0["pr_auc_ci"]["hi"], 4)])
        for metric, v in dev["care_lung_vs_device_only"].items():
            w.writerow(["paired_vs_device_only", f"delta_{metric}", round(v["delta"], 4),
                        round(v["lo"], 4), round(v["hi"], 4)])
        wm = dev["within_meditron"]
        for k in ["auroc", "pr_auc", "sensitivity", "specificity", "ece", "prevalence"]:
            ci = wm.get(f"{k}_ci")
            w.writerow(["within_meditron", k, round(wm[k], 4),
                        round(ci["lo"], 4) if ci else "", round(ci["hi"], 4) if ci else ""])
        for nm, v in dev["within_meditron_all_models"].items():
            w.writerow(["within_meditron_by_model", nm, round(v["auroc"], 4),
                        round(v["auroc_ci"]["lo"], 4), round(v["auroc_ci"]["hi"], 4)])
        for d_, v in dev["device_referral_rates"]["by_device"].items():
            w.writerow(["referral_rate_by_device", d_, round(v["refer_rate"], 4), "", ""])

    (OUT / "README.txt").write_text(
        "CARE-Lung audit artifacts\n"
        "=========================\n\n"
        "fold_split_manifest.csv   Patient identifiers for every within-fold subset\n"
        "                          (rf_train / calibration / threshold / test), with\n"
        "                          label and cycle count. Table 2 in the paper is a\n"
        "                          summary of this file.\n"
        "patient_predictions.csv   One row per patient: pooled out-of-fold score and\n"
        "                          hard prediction for all six aggregators and all six\n"
        "                          ablation variants, plus fold index and device.\n"
        "fold_thresholds.csv       Youden thresholds selected within each fold.\n"
        "pooled_metrics_with_ci.csv  Tables 3-5: pooled metrics with patient-grouped\n"
        "                          bootstrap 95% intervals (4,000 replicates).\n"
        "risk_coverage_curve.csv   Figure 4(a): selective risk vs. coverage.\n"
        "conformal_coverage.csv    Table 5 / Figure 2(b): marginal and Mondrian\n"
        "                          class-conditional coverage, referral, selective risk.\n"
        "device_confound_audit.csv Section 4.7: trivial and device-only baselines, the\n"
        "                          paired test against device-only, device-controlled\n"
        "                          (within-Meditron) metrics, and per-device referral.\n\n"
        "Regenerate with:\n"
        "  python scripts/run_revision_analyses.py\n"
        "  python scripts/run_device_audit.py\n"
        "  python scripts/export_audit_artifacts.py\n"
        "All seeds fixed (SEED=42); results are deterministic.\n",
        encoding="utf8")

    print(f"\nwrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
