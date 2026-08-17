"""Patient-level stratified 5-fold cross-validation for CARE-Lung and all
baseline aggregation strategies, evaluated under IDENTICAL out-of-fold
conditions.

Rationale: the original held-out test split has only 26 patients (5 healthy),
so single-split metrics such as accuracy / specificity swing on one or two
patients. This script reuses the SAME cached cycle features and the SAME
per-cycle model family (random forest + isotonic calibration) but evaluates
every aggregator (count-ratio, confidence-weighted, CARE-Lung, gradient-
boosted, SVM) with patient-level stratified 5-fold cross-validation, pooling
out-of-fold predictions over all 126 patients. No test-fold patient is ever
used for cycle-classifier training/calibration, aggregator fitting, or
threshold selection in the fold where it is scored -- this is a methodological
robustness upgrade, not a re-fit to the test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CACHE_FILE = RESULTS_DIR / "cycle_feature_cache.npz"
SEED = 42
N_FOLDS = 5

MODEL_NAMES = [
    "Count-ratio (BME106-style)",
    "Confidence-weighted fraction",
    "CARE-Lung (regularized distribution aggregator)",
    "Gradient-boosted distribution aggregator",
    "SVM + spectral summaries",
    "CARE-Lung++ (calibrated ensemble)",
]
PROPOSED = "CARE-Lung (regularized distribution aggregator)"
ENSEMBLE = "CARE-Lung++ (calibrated ensemble)"


def ece_score(y_true, prob, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(prob[mask]))
    return float(ece)


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def best_threshold(y_true, prob):
    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 181), prob])
    best_score, best_f1, best_t = -1.0, -1.0, 0.5
    for t in thresholds:
        pred = (prob >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        spec = specificity_score(y_true, pred)
        score = sens + spec - 1.0
        f1 = f1_score(y_true, pred, zero_division=0)
        if score > best_score or (score == best_score and f1 > best_f1):
            best_score, best_f1, best_t = score, f1, float(t)
    return best_t


def binary_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Specificity": specificity_score(y_true, pred),
        "ROC_AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "PR_AUC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ECE": ece_score(y_true, prob),
    }


def sigmoid_calibrate(raw_prob, fit_idx, y_fit):
    clipped = np.clip(raw_prob, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    cal.fit(logits[fit_idx], y_fit)
    return cal.predict_proba(logits)[:, 1]


def aggregate_patient_features(patients, devices, y_patient_cycle, prob, hard, wanted):
    rows, labels, ids = [], [], []
    wanted_set = set(wanted.tolist())
    for p in sorted(wanted_set):
        idx = patients == p
        p_prob, p_hard = prob[idx], hard[idx]
        entropy = -(p_prob * np.log(np.clip(p_prob, 1e-8, 1.0)) +
                    (1 - p_prob) * np.log(np.clip(1 - p_prob, 1e-8, 1.0))) / np.log(2)
        reliability = 1.0 - entropy
        weighted = float(np.sum(reliability * p_prob) / max(np.sum(reliability), 1e-8))
        rows.append([
            float(np.mean(p_hard)), float(np.mean(p_prob)), weighted, float(np.std(p_prob)),
            float(np.min(p_prob)), float(np.percentile(p_prob, 25)), float(np.median(p_prob)),
            float(np.percentile(p_prob, 75)), float(np.percentile(p_prob, 90)), float(np.max(p_prob)),
            float(np.mean(entropy)), float(np.sum(p_prob > 0.5)), float(np.log1p(np.sum(idx))),
            float(len(set(devices[idx].tolist()))),
        ])
        labels.append(int(y_patient_cycle[idx][0]))
        ids.append(p)
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=int), np.array(ids)


def main():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    X = cache["X"]
    patients = cache["patient"]
    devices = cache["device"]
    y_cycle = cache["y_cycle"].astype(int)
    y_patient_cycle = cache["y_patient"].astype(int)

    patient_ids_all = np.array(sorted(set(patients.tolist())))
    patient_labels = np.array([int(y_patient_cycle[patients == p][0]) for p in patient_ids_all])
    n_patients = len(patient_ids_all)
    print(f"Patients: {n_patients}  abnormal={int(patient_labels.sum())}  healthy={int((1 - patient_labels).sum())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pid_to_pos = {p: i for i, p in enumerate(patient_ids_all)}

    oof_prob = {name: np.full(n_patients, np.nan) for name in MODEL_NAMES}
    oof_pred = {name: np.full(n_patients, -1, dtype=int) for name in MODEL_NAMES}
    fold_metrics = {name: [] for name in MODEL_NAMES}

    for fold, (trainval_pos, test_pos) in enumerate(skf.split(patient_ids_all, patient_labels)):
        test_patients = patient_ids_all[test_pos]
        trainval_patients = patient_ids_all[trainval_pos]
        trainval_labels = patient_labels[trainval_pos]

        # inner split of the training-fold patients: 60% RF-train / 20% calibration / 20% threshold-selection
        rf_train_p, rest_p, rf_train_y, rest_y = train_test_split(
            trainval_patients, trainval_labels, test_size=0.40, random_state=SEED + fold, stratify=trainval_labels
        )
        cal_p, thr_p, cal_y, thr_y = train_test_split(
            rest_p, rest_y, test_size=0.5, random_state=SEED + 100 + fold, stratify=rest_y
        )

        def cmask(selected):
            sset = set(selected.tolist())
            return np.array([p in sset for p in patients])

        m_rf_train, m_cal = cmask(rf_train_p), cmask(cal_p)

        scaler = StandardScaler()
        rf = RandomForestClassifier(n_estimators=220, max_depth=12, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
        rf_pipe = make_pipeline(scaler, rf)
        rf_pipe.fit(X[m_rf_train], y_cycle[m_rf_train])
        calibrator = CalibratedClassifierCV(FrozenEstimator(rf_pipe), method="isotonic")
        calibrator.fit(X[m_cal], y_cycle[m_cal])
        all_prob = calibrator.predict_proba(X)[:, 1]
        all_hard = (all_prob >= 0.5).astype(int)

        agg_universe = np.concatenate([rf_train_p, cal_p, thr_p, test_patients])
        P, Py, Pids = aggregate_patient_features(patients, devices, y_patient_cycle, all_prob, all_hard, agg_universe)
        pos = {p: i for i, p in enumerate(Pids)}

        rf_train_idx = np.array([pos[p] for p in rf_train_p])
        cal_idx = np.array([pos[p] for p in cal_p])
        fit_idx = np.concatenate([rf_train_idx, cal_idx])           # 80% of trainval patients
        thr_idx = np.array([pos[p] for p in thr_p])                 # 20% of trainval patients
        test_idx = np.array([pos[p] for p in test_patients])        # held-out fold

        scores = {}
        scores["Count-ratio (BME106-style)"] = P[:, 0]
        scores["Confidence-weighted fraction"] = P[:, 2]

        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
        lr.fit(P[fit_idx], Py[fit_idx])
        scores["CARE-Lung (regularized distribution aggregator)"] = lr.predict_proba(P)[:, 1]

        gb = GradientBoostingClassifier(random_state=SEED, n_estimators=80, max_depth=2, learning_rate=0.06)
        gb.fit(P[rf_train_idx], Py[rf_train_idx])
        gb_raw = gb.predict_proba(P)[:, 1]
        scores["Gradient-boosted distribution aggregator"] = sigmoid_calibrate(gb_raw, cal_idx, Py[cal_idx])

        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                                                   probability=True, random_state=SEED))
        svm.fit(P[fit_idx], Py[fit_idx])
        scores["SVM + spectral summaries"] = svm.predict_proba(P)[:, 1]

        # CARE-Lung++ : average the two complementary, independently-fitted views
        # (regularized-linear ranking + RBF-kernel margin) into one calibrated score,
        # then re-fit a 1-D isotonic-style sigmoid map on the fit patients so the
        # ensemble output is itself a calibrated probability.
        ens_raw = 0.5 * scores["CARE-Lung (regularized distribution aggregator)"] + 0.5 * scores["SVM + spectral summaries"]
        scores["CARE-Lung++ (calibrated ensemble)"] = sigmoid_calibrate(ens_raw, fit_idx, Py[fit_idx])

        for name in MODEL_NAMES:
            prob_all = scores[name]
            threshold = best_threshold(Py[thr_idx], prob_all[thr_idx])
            pred_test = (prob_all[test_idx] >= threshold).astype(int)
            for p, pr, pd in zip(Pids[test_idx], prob_all[test_idx], pred_test):
                i = pid_to_pos[p]
                oof_prob[name][i] = pr
                oof_pred[name][i] = pd
            met = binary_metrics(Py[test_idx], prob_all[test_idx], threshold)
            met.update(fold=fold, n_test=int(len(test_idx)), threshold=threshold)
            fold_metrics[name].append(met)

        print(f"fold {fold} done (n_test={len(test_idx)}, "
              f"thr[CARE-Lung]={fold_metrics[PROPOSED][-1]['threshold']:.3f}, "
              f"F1[CARE-Lung]={fold_metrics[PROPOSED][-1]['F1']:.3f})")

    keys = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC_AUC", "PR_AUC", "ECE"]
    model_summaries = {}
    for name in MODEL_NAMES:
        assert not np.any(np.isnan(oof_prob[name]))
        pooled_pred = oof_pred[name]
        cm = confusion_matrix(patient_labels, pooled_pred, labels=[0, 1])
        pooled = {
            "Accuracy": float(accuracy_score(patient_labels, pooled_pred)),
            "Precision": float(precision_score(patient_labels, pooled_pred, zero_division=0)),
            "Recall": float(recall_score(patient_labels, pooled_pred, zero_division=0)),
            "F1": float(f1_score(patient_labels, pooled_pred, zero_division=0)),
            "Specificity": specificity_score(patient_labels, pooled_pred),
            "ROC_AUC": float(roc_auc_score(patient_labels, oof_prob[name])),
            "PR_AUC": float(average_precision_score(patient_labels, oof_prob[name])),
            "ECE": ece_score(patient_labels, oof_prob[name]),
        }
        fold_arr = {k: np.array([m[k] for m in fold_metrics[name]]) for k in keys}
        model_summaries[name] = {
            "pooled_out_of_fold_metrics": {k: round(v, 4) for k, v in pooled.items()},
            "pooled_confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
            "per_fold_mean_std": {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in fold_arr.items()},
        }
        print(f"\n=== {name} ===")
        print("pooled (n=126):", json.dumps(model_summaries[name]["pooled_out_of_fold_metrics"]))
        print("per-fold mean+-std:",
              {k: f"{v['mean']:.3f}+-{v['std']:.3f}" for k, v in model_summaries[name]["per_fold_mean_std"].items()})

    summary = {
        "n_folds": N_FOLDS,
        "n_patients": int(n_patients),
        "n_abnormal_patients": int(patient_labels.sum()),
        "n_healthy_patients": int((1 - patient_labels).sum()),
        "models": model_summaries,
    }
    (RESULTS_DIR / "cv_study_summary.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        RESULTS_DIR / "cv_eval_artifacts.npz",
        patient_ids=patient_ids_all,
        y_true=patient_labels,
        model_names=np.array(MODEL_NAMES, dtype=object),
        **{f"oof_prob__{i}": oof_prob[name] for i, name in enumerate(MODEL_NAMES)},
        **{f"oof_pred__{i}": oof_pred[name] for i, name in enumerate(MODEL_NAMES)},
    )
    print("\nwrote results/cv_study_summary.json and results/cv_eval_artifacts.npz")


if __name__ == "__main__":
    main()
