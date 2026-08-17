"""Reviewer-requested analyses for the CARE-Lung revision.

Adds, on top of the existing 5-fold patient-level protocol (unchanged, so all
headline numbers are reproduced exactly):

  (A) patient-grouped bootstrap confidence intervals for every pooled metric,
      plus paired bootstrap AUROC differences vs. CARE-Lung;
  (B) an explicit within-fold subset manifest (train / calibration / threshold
      / test patient counts and class balance per fold);
  (C) a pre-specified aggregator ablation: fixed, non-adaptive patient
      summaries of increasing dimensionality vs. the 14-D descriptor, under
      identical folds and identical fitting cohorts;
  (D) cross-conformal referral: marginal and class-conditional (Mondrian)
      coverage, referral rate, and selective-risk / coverage curves, with
      patient-level resampling;
  (E) device-stratified pooled out-of-fold errors over all 126 patients;
  (F) a variance decomposition explaining the fixed-split vs. pooled AUROC gap.
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
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
N_FOLDS = 5
N_BOOT = 4000

MODEL_NAMES = [
    "Count-ratio (BME106-style)",
    "Confidence-weighted fraction",
    "CARE-Lung (regularized distribution aggregator)",
    "Gradient-boosted distribution aggregator",
    "SVM + spectral summaries",
    "CARE-Lung++ (calibrated ensemble)",
]
PROPOSED = "CARE-Lung (regularized distribution aggregator)"

# --- (C) pre-specified, non-adaptive patient summaries -----------------------
# Column layout of the 14-D descriptor produced by aggregate_patient_features():
#  0 count-ratio  1 mean p  2 entropy-weighted  3 sd  4 min  5 p25  6 median
#  7 p75  8 p90  9 max  10 mean entropy  11 #high-conf abnormal
# 12 log cycle count  13 device count
ABLATIONS = {
    "S1: mean probability": [1],
    "S2: mean + SD": [1, 3],
    "S3: mean + SD + log cycles": [1, 3, 12],
    "S4: mean + SD + median + max": [1, 3, 6, 9],
    "S6: quantile summary": [1, 3, 4, 6, 9, 12],
    "S14: full descriptor (CARE-Lung)": list(range(14)),
}


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
        score = recall_score(y_true, pred, zero_division=0) + specificity_score(y_true, pred) - 1.0
        f1 = f1_score(y_true, pred, zero_division=0)
        if score > best_score or (score == best_score and f1 > best_f1):
            best_score, best_f1, best_t = score, f1, float(t)
    return best_t


def sigmoid_calibrate(raw_prob, fit_idx, y_fit):
    clipped = np.clip(raw_prob, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    cal.fit(logits[fit_idx], y_fit)
    return cal.predict_proba(logits)[:, 1]


def aggregate_patient_features(patients, devices, y_patient_cycle, prob, hard, wanted):
    rows, labels, ids = [], [], []
    for p in sorted(set(wanted.tolist())):
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


# ---------------------------------------------------------------- main sweep
def run_folds(X, patients, devices, y_cycle, y_patient_cycle, patient_ids_all, patient_labels):
    """Re-runs the published protocol and additionally scores the ablations."""
    n_patients = len(patient_ids_all)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pid_to_pos = {p: i for i, p in enumerate(patient_ids_all)}

    all_names = MODEL_NAMES + list(ABLATIONS)
    oof_prob = {n: np.full(n_patients, np.nan) for n in all_names}
    oof_pred = {n: np.full(n_patients, -1, dtype=int) for n in all_names}
    oof_fold = np.full(n_patients, -1, dtype=int)
    fold_metrics = {n: [] for n in all_names}
    manifest = []

    for fold, (trainval_pos, test_pos) in enumerate(skf.split(patient_ids_all, patient_labels)):
        test_patients = patient_ids_all[test_pos]
        trainval_patients = patient_ids_all[trainval_pos]
        trainval_labels = patient_labels[trainval_pos]

        rf_train_p, rest_p, rf_train_y, rest_y = train_test_split(
            trainval_patients, trainval_labels, test_size=0.40,
            random_state=SEED + fold, stratify=trainval_labels)
        cal_p, thr_p, cal_y, thr_y = train_test_split(
            rest_p, rest_y, test_size=0.5,
            random_state=SEED + 100 + fold, stratify=rest_y)

        def cmask(sel):
            s = set(sel.tolist())
            return np.array([p in s for p in patients])

        m_rf_train, m_cal = cmask(rf_train_p), cmask(cal_p)

        rf_pipe = make_pipeline(StandardScaler(),
                                RandomForestClassifier(n_estimators=220, max_depth=12,
                                                       class_weight="balanced",
                                                       random_state=SEED, n_jobs=-1))
        rf_pipe.fit(X[m_rf_train], y_cycle[m_rf_train])
        calibrator = CalibratedClassifierCV(FrozenEstimator(rf_pipe), method="isotonic")
        calibrator.fit(X[m_cal], y_cycle[m_cal])
        all_prob = calibrator.predict_proba(X)[:, 1]
        all_hard = (all_prob >= 0.5).astype(int)

        agg_universe = np.concatenate([rf_train_p, cal_p, thr_p, test_patients])
        P, Py, Pids = aggregate_patient_features(patients, devices, y_patient_cycle,
                                                 all_prob, all_hard, agg_universe)
        pos = {p: i for i, p in enumerate(Pids)}
        rf_train_idx = np.array([pos[p] for p in rf_train_p])
        cal_idx = np.array([pos[p] for p in cal_p])
        fit_idx = np.concatenate([rf_train_idx, cal_idx])
        thr_idx = np.array([pos[p] for p in thr_p])
        test_idx = np.array([pos[p] for p in test_patients])

        # ---- (B) subset manifest ------------------------------------------
        def counts(pp):
            lab = np.array([patient_labels[pid_to_pos[p]] for p in pp])
            cyc = int(sum(int(np.sum(patients == p)) for p in pp))
            return {"patients": int(len(pp)), "abnormal": int(lab.sum()),
                    "healthy": int((1 - lab).sum()), "cycles": cyc}

        manifest.append({
            "fold": fold,
            "rf_train": counts(rf_train_p), "calibration": counts(cal_p),
            "threshold": counts(thr_p), "test": counts(test_patients),
        })

        scores = {}
        scores["Count-ratio (BME106-style)"] = P[:, 0]
        scores["Confidence-weighted fraction"] = P[:, 2]

        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                max_iter=1000, random_state=SEED))
        lr.fit(P[fit_idx], Py[fit_idx])
        scores[PROPOSED] = lr.predict_proba(P)[:, 1]

        gb = GradientBoostingClassifier(random_state=SEED, n_estimators=80,
                                        max_depth=2, learning_rate=0.06)
        gb.fit(P[rf_train_idx], Py[rf_train_idx])
        scores["Gradient-boosted distribution aggregator"] = sigmoid_calibrate(
            gb.predict_proba(P)[:, 1], cal_idx, Py[cal_idx])

        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                  class_weight="balanced", probability=True,
                                                  random_state=SEED))
        svm.fit(P[fit_idx], Py[fit_idx])
        scores["SVM + spectral summaries"] = svm.predict_proba(P)[:, 1]

        ens_raw = 0.5 * scores[PROPOSED] + 0.5 * scores["SVM + spectral summaries"]
        scores["CARE-Lung++ (calibrated ensemble)"] = sigmoid_calibrate(ens_raw, fit_idx, Py[fit_idx])

        # ---- (C) pre-specified ablations, identical cohorts ----------------
        for name, cols in ABLATIONS.items():
            sub = make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=1000,
                                                   random_state=SEED))
            sub.fit(P[fit_idx][:, cols], Py[fit_idx])
            scores[name] = sub.predict_proba(P[:, cols])[:, 1]

        for name in all_names:
            prob_all = scores[name]
            thr = best_threshold(Py[thr_idx], prob_all[thr_idx])
            pred_test = (prob_all[test_idx] >= thr).astype(int)
            for p, pr, pd in zip(Pids[test_idx], prob_all[test_idx], pred_test):
                i = pid_to_pos[p]
                oof_prob[name][i] = pr
                oof_pred[name][i] = pd
                oof_fold[i] = fold
            yt = Py[test_idx]
            fold_metrics[name].append({
                "fold": fold, "threshold": thr,
                "Accuracy": float(accuracy_score(yt, pred_test)),
                "F1": float(f1_score(yt, pred_test, zero_division=0)),
                "Recall": float(recall_score(yt, pred_test, zero_division=0)),
                "Specificity": specificity_score(yt, pred_test),
                "ROC_AUC": float(roc_auc_score(yt, prob_all[test_idx])),
                "PR_AUC": float(average_precision_score(yt, prob_all[test_idx])),
                "ECE": ece_score(yt, prob_all[test_idx]),
            })
        print(f"  fold {fold}: n_test={len(test_idx)} "
              f"F1[CARE-Lung]={fold_metrics[PROPOSED][-1]['F1']:.3f}")

    return oof_prob, oof_pred, oof_fold, fold_metrics, manifest, all_names


# ------------------------------------------------------- (A) grouped bootstrap
def pooled_metrics(y, prob, pred):
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Specificity": specificity_score(y, pred),
        "AUROC": float(roc_auc_score(y, prob)),
        "PR_AUC": float(average_precision_score(y, prob)),
        "ECE": ece_score(y, prob),
    }


def grouped_bootstrap(y, prob, pred, n_boot=N_BOOT, seed=SEED):
    """Patient is the resampling unit: a drawn patient carries all of its
    cycles, its aggregated descriptor and its single pooled prediction."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["Accuracy", "Precision", "Sensitivity", "F1", "Specificity", "AUROC", "PR_AUC", "ECE"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = pooled_metrics(y[idx], prob[idx], pred[idx])
        for k in keys:
            draws[k].append(m[k])
    return {k: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for k, v in draws.items()}


def paired_auroc_delta(y, prob_a, prob_b, n_boot=N_BOOT, seed=SEED):
    """Paired patient-level bootstrap of AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], prob_a[idx]) - roc_auc_score(y[idx], prob_b[idx]))
    deltas = np.array(deltas)
    return {"delta": float(np.mean(deltas)),
            "lo": float(np.percentile(deltas, 2.5)),
            "hi": float(np.percentile(deltas, 97.5)),
            "p_gt_0": float(np.mean(deltas > 0))}


# --------------------------------------------------- (D) cross-conformal layer
def cross_conformal(y, prob, n_rep=400, cal_frac=0.5, alphas=(0.05, 0.10, 0.20, 0.30),
                    seed=SEED):
    """Repeated patient-level split-conformal over the pooled out-of-fold scores.

    Each replicate splits the 126 patients into a conformal-calibration half and
    an evaluation half, stratified by class so both halves keep healthy patients.
    Reports marginal coverage, class-conditional (Mondrian) coverage, referral
    rate, and selective risk = error rate among non-referred patients.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {a: {"cov": [], "cov_abn": [], "cov_hea": [], "refer": [],
               "refer_abn": [], "refer_hea": [], "sel_risk": [], "q": []} for a in alphas}
    out_mondrian = {a: {"cov_abn": [], "cov_hea": [], "refer": [], "sel_risk": []} for a in alphas}

    for _ in range(n_rep):
        cp = rng.permutation(pos)
        cn = rng.permutation(neg)
        cal = np.r_[cp[:int(len(cp) * cal_frac)], cn[:int(len(cn) * cal_frac)]]
        ev = np.r_[cp[int(len(cp) * cal_frac):], cn[int(len(cn) * cal_frac):]]
        # nonconformity: 1-p for abnormal, p for healthy
        s_cal = np.where(y[cal] == 1, 1.0 - prob[cal], prob[cal])
        for a in alphas:
            k = int(np.ceil((len(s_cal) + 1) * (1 - a)))
            qh = float(np.sort(s_cal)[min(k, len(s_cal)) - 1])
            in_pos = (1.0 - prob[ev]) <= qh          # label 1 in prediction set
            in_neg = prob[ev] <= qh                  # label 0 in prediction set
            refer = in_pos & in_neg
            covered = np.where(y[ev] == 1, in_pos, in_neg)
            decided = ~refer & (in_pos | in_neg)
            if decided.sum():
                sel_risk = float(np.mean(in_pos[decided].astype(int) != y[ev][decided]))
            else:
                sel_risk = np.nan
            o = out[a]
            o["q"].append(qh)
            o["cov"].append(float(np.mean(covered)))
            o["cov_abn"].append(float(np.mean(covered[y[ev] == 1])))
            o["cov_hea"].append(float(np.mean(covered[y[ev] == 0])))
            o["refer"].append(float(np.mean(refer)))
            o["refer_abn"].append(float(np.mean(refer[y[ev] == 1])))
            o["refer_hea"].append(float(np.mean(refer[y[ev] == 0])))
            o["sel_risk"].append(sel_risk)

            # class-conditional (Mondrian) conformal: separate quantile per class
            s_abn = 1.0 - prob[cal][y[cal] == 1]
            s_hea = prob[cal][y[cal] == 0]

            def q_of(s):
                kk = int(np.ceil((len(s) + 1) * (1 - a)))
                return float(np.sort(s)[min(kk, len(s)) - 1])

            q_a, q_h = q_of(s_abn), q_of(s_hea)
            mp = (1.0 - prob[ev]) <= q_a
            mn = prob[ev] <= q_h
            mrefer = mp & mn
            mcov = np.where(y[ev] == 1, mp, mn)
            mdec = ~mrefer & (mp | mn)
            om = out_mondrian[a]
            om["cov_abn"].append(float(np.mean(mcov[y[ev] == 1])))
            om["cov_hea"].append(float(np.mean(mcov[y[ev] == 0])))
            om["refer"].append(float(np.mean(mrefer)))
            om["sel_risk"].append(float(np.mean(mp[mdec].astype(int) != y[ev][mdec]))
                                  if mdec.sum() else np.nan)

    def summarise(d):
        return {k: {"mean": float(np.nanmean(v)),
                    "lo": float(np.nanpercentile(v, 2.5)),
                    "hi": float(np.nanpercentile(v, 97.5))} for k, v in d.items()}

    return ({str(a): summarise(v) for a, v in out.items()},
            {str(a): summarise(v) for a, v in out_mondrian.items()})


def risk_coverage_curve(y, prob, thr, grid=None):
    """Confidence-based selective prediction: sweep the abstention band around
    the operating threshold and report retained coverage vs. error on retained."""
    if grid is None:
        grid = np.linspace(0.0, 0.45, 46)
    pred = (prob >= thr).astype(int)
    conf = np.abs(prob - thr)
    rows = []
    for b in grid:
        keep = conf >= b
        if keep.sum() < 5 or len(np.unique(y[keep])) < 2:
            break
        rows.append({"band": float(b), "coverage": float(np.mean(keep)),
                     "risk": float(np.mean(pred[keep] != y[keep])),
                     "sens": float(recall_score(y[keep], pred[keep], zero_division=0)),
                     "spec": specificity_score(y[keep], pred[keep])})
    return rows


def main():
    cache = np.load(RESULTS / "cycle_feature_cache.npz", allow_pickle=True)
    X, patients, devices = cache["X"], cache["patient"], cache["device"]
    y_cycle = cache["y_cycle"].astype(int)
    y_patient_cycle = cache["y_patient"].astype(int)
    patient_ids_all = np.array(sorted(set(patients.tolist())))
    patient_labels = np.array([int(y_patient_cycle[patients == p][0]) for p in patient_ids_all])

    print("Re-running 5-fold protocol with ablations ...")
    oof_prob, oof_pred, oof_fold, fold_metrics, manifest, all_names = run_folds(
        X, patients, devices, y_cycle, y_patient_cycle, patient_ids_all, patient_labels)

    y = patient_labels
    out = {"n_patients": int(len(y)), "n_abnormal": int(y.sum()),
           "n_healthy": int((1 - y).sum()), "fold_manifest": manifest}

    # (A) pooled metrics + grouped bootstrap CIs
    print("\nGrouped bootstrap ...")
    models = {}
    for name in all_names:
        p, pr = oof_prob[name], oof_pred[name]
        pooled = pooled_metrics(y, p, pr)
        ci = grouped_bootstrap(y, p, pr)
        fm = {k: {"mean": float(np.mean([m[k] for m in fold_metrics[name]])),
                  "std": float(np.std([m[k] for m in fold_metrics[name]])),
                  "min": float(np.min([m[k] for m in fold_metrics[name]])),
                  "max": float(np.max([m[k] for m in fold_metrics[name]]))}
              for k in ["Accuracy", "F1", "Recall", "Specificity", "ROC_AUC", "PR_AUC", "ECE"]}
        models[name] = {"pooled": {k: round(v, 4) for k, v in pooled.items()},
                        "bootstrap_ci95": {k: {"lo": round(v["lo"], 4), "hi": round(v["hi"], 4)}
                                           for k, v in ci.items()},
                        "per_fold": fm,
                        "fold_thresholds": [round(m["threshold"], 4) for m in fold_metrics[name]]}
        print(f"  {name[:44]:44s} AUROC={pooled['AUROC']:.3f} "
              f"[{ci['AUROC']['lo']:.3f},{ci['AUROC']['hi']:.3f}]  F1={pooled['F1']:.3f}")
    out["models"] = models

    # paired AUROC deltas vs CARE-Lung
    print("\nPaired AUROC deltas vs CARE-Lung ...")
    deltas = {}
    for name in all_names:
        if name == PROPOSED:
            continue
        deltas[name] = paired_auroc_delta(y, oof_prob[PROPOSED], oof_prob[name])
        d = deltas[name]
        print(f"  vs {name[:40]:40s} d={d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
              f"P(d>0)={d['p_gt_0']:.3f}")
    out["paired_auroc_delta_vs_care_lung"] = deltas

    # (D) conformal
    print("\nCross-conformal referral ...")
    marg, mond = cross_conformal(y, oof_prob[PROPOSED])
    out["cross_conformal_marginal"] = marg
    out["cross_conformal_mondrian"] = mond
    for a in marg:
        m = marg[a]
        print(f"  alpha={a}: cov={m['cov']['mean']:.3f} "
              f"cov_abn={m['cov_abn']['mean']:.3f} cov_hea={m['cov_hea']['mean']:.3f} "
              f"refer={m['refer']['mean']:.3f} sel_risk={m['sel_risk']['mean']:.3f}")
    for a in mond:
        m = mond[a]
        print(f"  MONDRIAN alpha={a}: cov_abn={m['cov_abn']['mean']:.3f} "
              f"cov_hea={m['cov_hea']['mean']:.3f} refer={m['refer']['mean']:.3f} "
              f"sel_risk={m['sel_risk']['mean']:.3f}")

    med_thr = float(np.median([m["threshold"] for m in fold_metrics[PROPOSED]]))
    out["median_fold_threshold"] = med_thr
    out["risk_coverage"] = risk_coverage_curve(y, oof_prob[PROPOSED], med_thr)

    # (E) device-stratified pooled OOF
    print("\nDevice-stratified pooled out-of-fold (CARE-Lung) ...")
    pdev = {}
    for p in patient_ids_all:
        d, c = np.unique(devices[patients == p], return_counts=True)
        pdev[p] = d[np.argmax(c)]
    dev_arr = np.array([pdev[p] for p in patient_ids_all])
    dev_stats = {}
    for d in sorted(set(dev_arr.tolist())):
        m = dev_arr == d
        yy, pp, rr = y[m], oof_prob[PROPOSED][m], oof_pred[PROPOSED][m]
        dev_stats[d] = {
            "n": int(m.sum()), "n_abnormal": int(yy.sum()), "n_healthy": int((1 - yy).sum()),
            "sensitivity": float(recall_score(yy, rr, zero_division=0)) if yy.sum() else None,
            "specificity": specificity_score(yy, rr) if (1 - yy).sum() else None,
            "accuracy": float(accuracy_score(yy, rr)),
            "AUROC": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else None,
            "mean_score": float(np.mean(pp)),
        }
        print(f"  {d:10s} n={m.sum():3d} ({int(yy.sum())}/{int((1-yy).sum())}) "
              f"acc={dev_stats[d]['accuracy']:.3f} sens={dev_stats[d]['sensitivity']} "
              f"spec={dev_stats[d]['specificity']}")
    out["device_stratified_oof"] = dev_stats

    # (F) fixed-split variance: bootstrap the 26-patient split for comparison
    print("\nFixed-split AUROC uncertainty ...")
    ev = np.load(RESULTS / "eval_artifacts.npz", allow_pickle=True)
    names_fixed = [str(n) for n in ev["model_names"]]
    ci_fixed = None
    if PROPOSED in names_fixed:
        j = names_fixed.index(PROPOSED)
        yf, pf = ev["y_test_patient"], ev["score_matrix_test"][j]
        rng = np.random.default_rng(SEED)
        vals = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(yf), len(yf))
            if len(np.unique(yf[idx])) < 2:
                continue
            vals.append(roc_auc_score(yf[idx], pf[idx]))
        ci_fixed = {"auroc": float(roc_auc_score(yf, pf)),
                    "lo": float(np.percentile(vals, 2.5)),
                    "hi": float(np.percentile(vals, 97.5)),
                    "n_test": int(len(yf)), "n_healthy": int((1 - yf).sum()),
                    "n_discordant_pairs": int(yf.sum() * (1 - yf).sum())}
        print(f"  fixed split AUROC={ci_fixed['auroc']:.3f} "
              f"[{ci_fixed['lo']:.3f},{ci_fixed['hi']:.3f}] "
              f"on {ci_fixed['n_discordant_pairs']} discordant pairs")
    out["fixed_split_auroc"] = ci_fixed
    out["auroc_gap_note"] = {
        "pooled": models[PROPOSED]["pooled"]["AUROC"],
        "pooled_ci": models[PROPOSED]["bootstrap_ci95"]["AUROC"],
        "pooled_discordant_pairs": int(y.sum() * (1 - y).sum()),
    }

    (RESULTS / "revision_analyses.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(RESULTS / "revision_oof.npz",
                        patient_ids=patient_ids_all, y_true=y, fold=oof_fold,
                        device=dev_arr, model_names=np.array(all_names, dtype=object),
                        **{f"prob__{i}": oof_prob[n] for i, n in enumerate(all_names)},
                        **{f"pred__{i}": oof_pred[n] for i, n in enumerate(all_names)})
    print("\nwrote results/revision_analyses.json and results/revision_oof.npz")


if __name__ == "__main__":
    main()
