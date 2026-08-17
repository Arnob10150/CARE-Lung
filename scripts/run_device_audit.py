"""Device-confound audit for CARE-Lung.

The ICBHI cohort confounds recording device with class: all 26 healthy patients
were recorded with a Meditron stethoscope. Reporting device-stratified errors
(as the paper already does) shows that errors differ by device, but it does not
answer the question a sceptical reader actually has -- how much of the reported
patient-level discrimination is attributable to the device rather than to
respiratory acoustics?

This script answers it with three controls that need no additional data:

  (A) Device-only baseline. Predict the patient label from the stethoscope
      identity alone, scored strictly out-of-fold on the same 5 folds. Whatever
      AUROC this reaches is available without listening to the audio at all.
  (B) Paired comparison. Patient-grouped paired bootstrap of
      AUROC(CARE-Lung) - AUROC(device-only), to test whether the acoustic
      pipeline adds anything beyond device identity.
  (C) Device-controlled evaluation. Restrict to Meditron, the only cohort
      containing both classes, so device is held constant by construction.
      Report full metrics, and re-run the referral analysis there to test
      whether the calibration/abstention machinery survives the control.

Outputs results/device_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
N_BOOT = 4000
PROPOSED = "CARE-Lung (regularized distribution aggregator)"


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def ece_score(y_true, prob, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if not np.any(m):
            continue
        ece += np.mean(m) * abs(np.mean(y_true[m]) - np.mean(prob[m]))
    return float(ece)


def boot_ci(fn, y, *arrays, n_boot=N_BOOT, seed=SEED):
    """Patient-grouped bootstrap percentile interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(fn(y[idx], *[a[idx] for a in arrays]))
        except ValueError:
            continue
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def paired_delta(y, a, b, fn=roc_auc_score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    d = np.array(d)
    return {"delta": float(np.mean(d)), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "p_gt_0": float(np.mean(d > 0))}


def risk_coverage(y, prob, thr, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 0.45, 46)
    pred = (prob >= thr).astype(int)
    conf = np.abs(prob - thr)
    rows = []
    for b in grid:
        keep = conf >= b
        if keep.sum() < 8 or len(np.unique(y[keep])) < 2:
            break
        rows.append({"band": float(b), "coverage": float(np.mean(keep)),
                     "risk": float(np.mean(pred[keep] != y[keep])),
                     "sens": float(recall_score(y[keep], pred[keep], zero_division=0))})
    return rows


def main():
    o = np.load(RESULTS / "revision_oof.npz", allow_pickle=True)
    names = [str(n) for n in o["model_names"]]
    j = names.index(PROPOSED)
    y = o["y_true"].astype(int)
    prob = o[f"prob__{j}"]
    pred = o[f"pred__{j}"].astype(int)
    dev = o["device"]
    pid = o["patient_ids"]
    n = len(y)

    out = {"n": int(n), "n_abnormal": int(y.sum()), "n_healthy": int((1 - y).sum())}

    # ---- trivial majority baseline -----------------------------------------
    prev = float(y.mean())
    out["trivial_all_abnormal"] = {
        "accuracy": prev, "precision": prev, "sensitivity": 1.0,
        "specificity": 0.0, "f1": float(2 * prev / (1 + prev)),
        "auroc": 0.5, "pr_auc": prev,
    }

    # ---- (A) device-only baseline, strictly out-of-fold --------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_score = np.full(n, np.nan)
    for tr, te in skf.split(pid, y):
        rates = {}
        for d in set(dev.tolist()):
            m = dev[tr] == d
            rates[d] = float(y[tr][m].mean()) if m.sum() else float(y[tr].mean())
        for i in te:
            dev_score[i] = rates[dev[i]]
    assert not np.isnan(dev_score).any()

    out["device_only_baseline"] = {
        "auroc": float(roc_auc_score(y, dev_score)),
        "auroc_ci": boot_ci(roc_auc_score, y, dev_score),
        "pr_auc": float(average_precision_score(y, dev_score)),
        "pr_auc_ci": boot_ci(average_precision_score, y, dev_score),
    }

    # ---- (B) does the acoustic pipeline beat device identity? --------------
    out["care_lung_vs_device_only"] = {
        "auroc": paired_delta(y, prob, dev_score, roc_auc_score),
        "pr_auc": paired_delta(y, prob, dev_score, average_precision_score),
    }

    # also: how much does device explain of each aggregator?
    per_model = {}
    for k, nm in enumerate(names):
        p = o[f"prob__{k}"]
        per_model[nm] = {
            "auroc": float(roc_auc_score(y, p)),
            "delta_vs_device_only": paired_delta(y, p, dev_score, roc_auc_score),
        }
    out["all_models_vs_device_only"] = per_model

    # ---- (C) device-controlled evaluation: Meditron only -------------------
    m = dev == "Meditron"
    yy, pp, rr = y[m], prob[m], pred[m]
    thr_med = 0.495  # median fold threshold, as used for the pooled analysis
    out["within_meditron"] = {
        "n": int(m.sum()), "n_abnormal": int(yy.sum()), "n_healthy": int((1 - yy).sum()),
        "prevalence": float(yy.mean()),
        "accuracy": float(accuracy_score(yy, rr)),
        "precision": float(precision_score(yy, rr, zero_division=0)),
        "sensitivity": float(recall_score(yy, rr, zero_division=0)),
        "specificity": specificity_score(yy, rr),
        "f1": float(f1_score(yy, rr, zero_division=0)),
        "auroc": float(roc_auc_score(yy, pp)),
        "auroc_ci": boot_ci(roc_auc_score, yy, pp),
        "pr_auc": float(average_precision_score(yy, pp)),
        "pr_auc_ci": boot_ci(average_precision_score, yy, pp),
        "ece": ece_score(yy, pp),
        "ece_ci": boot_ci(lambda a, b: ece_score(a, b), yy, pp),
    }
    # trivial baseline inside Meditron
    pv = float(yy.mean())
    out["within_meditron_trivial"] = {
        "accuracy": pv, "f1": float(2 * pv / (1 + pv)), "auroc": 0.5, "pr_auc": pv}

    # all aggregators within Meditron, for a fair device-controlled ranking
    wm = {}
    for k, nm in enumerate(names):
        p = o[f"prob__{k}"][m]
        pr_ = o[f"pred__{k}"][m].astype(int)
        wm[nm] = {"auroc": float(roc_auc_score(yy, p)),
                  "auroc_ci": boot_ci(roc_auc_score, yy, p),
                  "pr_auc": float(average_precision_score(yy, p)),
                  "sensitivity": float(recall_score(yy, pr_, zero_division=0)),
                  "specificity": specificity_score(yy, pr_),
                  "ece": ece_score(yy, p)}
    out["within_meditron_all_models"] = wm

    # ---- does the referral layer survive the device control? ---------------
    out["within_meditron_risk_coverage"] = risk_coverage(yy, pp, thr_med)
    out["pooled_risk_coverage_ref"] = risk_coverage(y, prob, thr_med)

    # ---- device-stratified referral behaviour on the full cohort -----------
    # confidence-band abstention at a fixed band, per device
    band = 0.20
    conf = np.abs(prob - thr_med)
    refer = conf < band
    dev_ref = {}
    for d in sorted(set(dev.tolist())):
        md = dev == d
        dev_ref[d] = {"n": int(md.sum()),
                      "refer_rate": float(np.mean(refer[md])),
                      "mean_score": float(np.mean(prob[md]))}
    out["device_referral_rates"] = {"band": band, "by_device": dev_ref}

    (RESULTS / "device_audit.json").write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- report
    d0 = out["device_only_baseline"]
    cv = out["care_lung_vs_device_only"]["auroc"]
    wmm = out["within_meditron"]
    print("=" * 68)
    print("(A) DEVICE-ONLY BASELINE (no audio, out-of-fold)")
    print(f"    AUROC  {d0['auroc']:.3f} [{d0['auroc_ci']['lo']:.3f}, {d0['auroc_ci']['hi']:.3f}]")
    print(f"    PR-AUC {d0['pr_auc']:.3f} [{d0['pr_auc_ci']['lo']:.3f}, {d0['pr_auc_ci']['hi']:.3f}]")
    print()
    print("(B) CARE-Lung MINUS device-only (paired, patient-grouped)")
    print(f"    dAUROC {cv['delta']:+.3f} [{cv['lo']:+.3f}, {cv['hi']:+.3f}]  P(d>0)={cv['p_gt_0']:.3f}")
    print()
    print("(C) DEVICE-CONTROLLED (Meditron only, n=%d: %d abn / %d healthy)"
          % (wmm["n"], wmm["n_abnormal"], wmm["n_healthy"]))
    print(f"    AUROC  {wmm['auroc']:.3f} [{wmm['auroc_ci']['lo']:.3f}, {wmm['auroc_ci']['hi']:.3f}]")
    print(f"    PR-AUC {wmm['pr_auc']:.3f} (prevalence {wmm['prevalence']:.3f})")
    print(f"    Sens {wmm['sensitivity']:.3f}  Spec {wmm['specificity']:.3f}  ECE {wmm['ece']:.3f}")
    print()
    print("    within-Meditron ranking:")
    for nm, v in wm.items():
        print(f"      {nm[:44]:44s} AUROC {v['auroc']:.3f} "
              f"[{v['auroc_ci']['lo']:.3f},{v['auroc_ci']['hi']:.3f}]  ECE {v['ece']:.3f}")
    print()
    rc = out["within_meditron_risk_coverage"]
    print("    referral inside Meditron (device held constant):")
    for r in rc[::6]:
        print(f"      coverage {r['coverage']:.3f}  risk {r['risk']:.3f}  sens {r['sens']:.3f}")
    print()
    print("    device-wise referral rate at band %.2f:" % band)
    for d, v in dev_ref.items():
        print(f"      {d:10s} n={v['n']:3d}  refer {v['refer_rate']:.3f}  mean score {v['mean_score']:.3f}")
    print("=" * 68)
    print("wrote results/device_audit.json")


if __name__ == "__main__":
    main()
