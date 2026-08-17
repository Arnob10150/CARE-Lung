"""Generate every paper figure to Springer camera-ready specification.

Two hard requirements from the AII 2026 / Springer instructions drive this file:

  * maximum figure width 12 cm (= 4.724 in);
  * no text smaller than 7 pt *in the final rescaled figure*.

Both are met by designing each figure at exactly the width it is placed at in
the document and never rescaling it in LaTeX, so the point sizes written here
are the point sizes that reach the page. Every figure is included with
`width=<design width>in`, which matches its natural size.

Output names follow the required convention Fig01-44.pdf ... Fig09-44.pdf.
All figures are vector PDF generated from the released result files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "camera_ready_figures"
OUT.mkdir(exist_ok=True)

MAXW = 4.45          # inches; tight-bbox padding lands the output just under 12 cm
BASE = 7.5           # nothing on any axis may fall below 7 pt

plt.rcParams.update({
    "font.size": BASE, "font.family": "serif",
    "axes.titlesize": 8.0, "axes.labelsize": BASE,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

C_PROP, C_ENS = "#dc2626", "#d97706"
C_ABN, C_HEA, C_GREY = "#1d4ed8", "#059669", "#9aa3af"

SHORT = {
    "Count-ratio (BME106-style)": "Count-ratio",
    "Confidence-weighted fraction": "Conf.-wtd",
    "CARE-Lung (regularized distribution aggregator)": "CARE-Lung",
    "Gradient-boosted distribution aggregator": "Grad.-boost",
    "SVM + spectral summaries": "SVM summ.",
    "CARE-Lung++ (calibrated ensemble)": "CARE-Lung++",
}
COLORS = {
    "Count-ratio (BME106-style)": "#9aa3af",
    "Confidence-weighted fraction": "#3b82f6",
    "CARE-Lung (regularized distribution aggregator)": "#dc2626",
    "Gradient-boosted distribution aggregator": "#8b5cf6",
    "SVM + spectral summaries": "#059669",
    "CARE-Lung++ (calibrated ensemble)": "#d97706",
}
PROP = "CARE-Lung (regularized distribution aggregator)"


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------- Fig 1
def fig01_reliability():
    art = np.load(RESULTS / "eval_artifacts.npz", allow_pickle=True)
    y = art["y_cycle_test"].astype(int)
    before, after = art["uncal_cycle_prob_test"], art["cal_cycle_prob_test"]

    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.48))
    bins = np.linspace(0, 1, 9)
    for prob, lab, col, mk in [(before, "Uncalibrated", C_GREY, "o"),
                               (after, "Isotonic-calibrated", C_PROP, "s")]:
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
            if np.any(m):
                xs.append(float(np.mean(prob[m]))); ys.append(float(np.mean(y[m])))
                ns.append(int(m.sum()))
        ece = float(np.sum(np.array(ns) / len(prob) * np.abs(np.array(ys) - np.array(xs))))
        ax[0].plot(xs, ys, marker=mk, ms=3.4, lw=1.4, color=col, label=f"{lab} ({ece:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6, label="Perfect")
    ax[0].set_xlabel("Mean predicted probability")
    ax[0].set_ylabel("Empirical abnormal rate")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_title("(a) Reliability curve")
    ax[0].legend(frameon=False, loc="upper left", handlelength=1.5)

    ax[1].hist(before, bins=bins, alpha=0.55, color=C_GREY, label="Uncalibrated", density=True)
    ax[1].hist(after, bins=bins, alpha=0.55, color=C_PROP, label="Calibrated", density=True)
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Density")
    ax[1].set_title("(b) Score distribution")
    ax[1].legend(frameon=False, loc="upper right", handlelength=1.5)
    fig.tight_layout(pad=0.3)
    save(fig, "Fig01-44.pdf")


# --------------------------------------------------------------- Fig 2
def fig02_architecture():
    """Pipeline diagram. Text kept short so every label clears 7 pt at 11.94 cm."""
    fig, ax = plt.subplots(figsize=(MAXW, 2.30))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(cx, cy, w, h, title, detail, fc):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=0.9",
                                    lw=0.7, edgecolor="#374151", facecolor=fc, zorder=2))
        ax.text(cx, cy + (1.9 if detail else 0), title, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color="#111827", zorder=3)
        if detail:
            ax.text(cx, cy - 2.4, detail, ha="center", va="center", fontsize=7.0,
                    color="#1f2937", zorder=3, linespacing=1.25)
        return (cx, cy, w, h)

    def arrow(b1, b2, s, e, rad=0.0):
        x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
        P = {"r": (x1 + w1 / 2, y1), "l": (x1 - w1 / 2, y1),
             "t": (x1, y1 + h1 / 2), "b": (x1, y1 - h1 / 2)}
        Q = {"r": (x2 + w2 / 2, y2), "l": (x2 - w2 / 2, y2),
             "t": (x2, y2 + h2 / 2), "b": (x2, y2 - h2 / 2)}
        ax.add_patch(FancyArrowPatch(P[s], Q[e], arrowstyle="-|>", mutation_scale=6,
                                     lw=0.8, color="#4b5563", zorder=1,
                                     shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f"arc3,rad={rad}"))

    y1, y2, y3 = 50, 30, 9
    ax.text(0, 61.5, "(a)  Per-cycle calibration", color="#1e3a8a", va="top", fontsize=7.4, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    a1 = box(12.5, y1, 23, 9.5, "Cycle audio", "ICBHI segment", "#eef2ff")
    a2 = box(37.5, y1, 23, 9.5, "31-d features", "spectral, MFCC", "#e0e7ff")
    a3 = box(62.5, y1, 23, 9.5, "Random forest", "class-balanced", "#dbeafe")
    a4 = box(87.5, y1, 23, 9.5, "Isotonic calib.", "ECE .198 to .147", "#cffafe")
    for p, q in [(a1, a2), (a2, a3), (a3, a4)]:
        arrow(p, q, "r", "l")

    ax.text(0, 41.5, "(b)  Cycle-to-patient aggregation", color="#92400e", va="top", fontsize=7.4, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    b1 = box(87.5, y2, 23, 9.5, "Patient summary", "14-d descriptor", "#fde68a")
    b2 = box(55.0, y2, 35, 9.5, "CARE-Lung aggregator", "logistic regression", "#fecaca")
    b3 = box(15.5, y2, 27, 9.5, "Patient score", "AUROC 0.832", "#fee2e2")
    arrow(a4, b1, "b", "t")
    arrow(b1, b2, "l", "r"); arrow(b2, b3, "l", "r")

    ax.text(0, 20.5, "(c)  Decision and conformal triage", color="#065f46", va="top", fontsize=7.4, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    c1 = box(25.0, y3, 44, 10.5, "Operating point", "Youden J, held-out split", "#dcfce7")
    c2 = box(74.0, y3, 46, 10.5, "Conformal referral", "screen +/-, or refer", "#bbf7d0")
    arrow(b3, c1, "b", "t", rad=0.18); arrow(b3, c2, "b", "t", rad=-0.14)

    fig.tight_layout(pad=0.2)
    save(fig, "Fig02-44.pdf")


# --------------------------------------------------------------- Fig 3
def fig03_rocpr():
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    art = np.load(RESULTS / "cv_eval_artifacts.npz", allow_pickle=True)
    names = [str(n) for n in art["model_names"]]
    y = art["y_true"].astype(int)
    prob = {n: art[f"oof_prob__{i}"] for i, n in enumerate(names)}

    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.54))
    for n in SHORT:
        fpr, tpr, _ = roc_curve(y, prob[n])
        lw = 1.9 if n == PROP else 1.1
        ax[0].plot(fpr, tpr, color=COLORS[n], lw=lw, zorder=5 if n == PROP else 2,
                   label=f"{SHORT[n]} ({roc_auc_score(y, prob[n]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.45)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) Pooled out-of-fold ROC")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.04)
    ax[0].legend(frameon=False, loc="lower right", fontsize=7.0, handlelength=1.3,
                 labelspacing=0.22, borderpad=0.15)

    prev = float(np.mean(y))
    for n in SHORT:
        pr, rc, _ = precision_recall_curve(y, prob[n])
        lw = 1.9 if n == PROP else 1.1
        ax[1].plot(rc, pr, color=COLORS[n], lw=lw, zorder=5 if n == PROP else 2)
    ax[1].axhline(prev, color="k", ls="--", lw=0.8, alpha=0.45)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("(b) Precision--recall")
    ax[1].set_xlim(-0.02, 1.02); ax[1].set_ylim(0.4, 1.04)
    fig.tight_layout(pad=0.3)
    save(fig, "Fig03-44.pdf")


# --------------------------------------------------------------- Fig 4
def fig04_confusion(rev, o):
    names = [str(n) for n in o["model_names"]]
    j = names.index(PROP)
    y, pred = o["y_true"].astype(int), o[f"pred__{j}"].astype(int)
    cm = np.array([[int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
                   [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]])

    fig, a = plt.subplots(figsize=(2.55, 1.48))
    a.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Reds", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            f = cm[r, c] / cm[r].sum()
            a.text(c, r, f"{cm[r, c]}\n({f:.2f})", ha="center", va="center",
                   fontsize=7.6, fontweight="bold", color="white" if f > 0.55 else "#111827")
    a.set_xticks([0, 1], ["pred.\nhealthy", "pred.\nabnormal"])
    a.set_yticks([0, 1], ["healthy", "abnormal"])
    a.grid(False)
    fig.tight_layout(pad=0.25)
    save(fig, "Fig04-44.pdf")


# --------------------------------------------------------------- Fig 5
def fig05_calibration(rev, o):
    order = list(SHORT)
    ece = [rev["models"][n]["pooled"]["ECE"] for n in order]
    lo = [rev["models"][n]["bootstrap_ci95"]["ECE"]["lo"] for n in order]
    hi = [rev["models"][n]["bootstrap_ci95"]["ECE"]["hi"] for n in order]
    xs = np.arange(len(order))
    err = np.vstack([np.array(ece) - np.array(lo), np.array(hi) - np.array(ece)])

    fig, a = plt.subplots(figsize=(MAXW, 1.48))
    a.bar(xs, ece, color=[COLORS[n] for n in order], width=0.66,
          yerr=err, error_kw=dict(lw=0.7, capsize=2.0, ecolor="#374151"))
    a.set_xticks(xs, [SHORT[n] for n in order], rotation=18, ha="right")
    a.set_ylabel("Expected calibration error")
    fig.tight_layout(pad=0.25)
    save(fig, "Fig05-44.pdf")


# --------------------------------------------------------------- Fig 6
def fig06_ensemble(rev, o):
    keys = ["Sensitivity", "Specificity", "Precision", "F1", "AUROC"]
    p1 = rev["models"][PROP]["pooled"]
    p2 = rev["models"]["CARE-Lung++ (calibrated ensemble)"]["pooled"]
    xs = np.arange(len(keys)); w = 0.36

    fig, a = plt.subplots(figsize=(3.40, 1.52))
    a.bar(xs - w / 2, [p1[k] for k in keys], w, color=C_PROP, label="CARE-Lung")
    a.bar(xs + w / 2, [p2[k] for k in keys], w, color=C_ENS, label="CARE-Lung++")
    a.set_xticks(xs, ["Sens.", "Spec.", "Prec.", "F1", "AUROC"])
    a.set_ylim(0, 1.36); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.3, columnspacing=1.0)
    fig.tight_layout(pad=0.25)
    save(fig, "Fig06-44.pdf")


# --------------------------------------------------------------- Fig 7
def fig07_conformal(rev, o):
    marg, mond = rev["cross_conformal_marginal"], rev["cross_conformal_mondrian"]
    al = sorted(marg, key=float); xs = np.array([float(x) for x in al])

    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.48))
    a = ax[0]
    a.plot(xs, 1 - xs, "k:", lw=0.9, label="Target $1-\\alpha$")
    a.plot(xs, [marg[x]["cov_abn"]["mean"] for x in al], "-o", color=C_ABN, ms=3.2, lw=1.2, label="Marg., abn.")
    a.plot(xs, [marg[x]["cov_hea"]["mean"] for x in al], "-o", color=C_HEA, ms=3.2, lw=1.2, label="Marg., healthy")
    a.plot(xs, [mond[x]["cov_abn"]["mean"] for x in al], "--^", color=C_ABN, ms=3.2, lw=1.1, alpha=.85, label="Mond., abn.")
    a.plot(xs, [mond[x]["cov_hea"]["mean"] for x in al], "--^", color=C_HEA, ms=3.2, lw=1.1, alpha=.85, label="Mond., healthy")
    a.set_xlabel("Target error level $\\alpha$"); a.set_ylabel("Empirical coverage")
    a.set_xticks(xs); a.set_ylim(0.60, 1.06)
    a.set_title("(a) Class-conditional coverage")
    a.legend(frameon=False, loc="lower left", fontsize=7.0, handlelength=1.3, labelspacing=0.2)

    a = ax[1]
    a.plot(xs, [marg[x]["refer"]["mean"] for x in al], "-o", color=C_PROP, ms=3.2, lw=1.2, label="Refer, marg.")
    a.plot(xs, [mond[x]["refer"]["mean"] for x in al], "--^", color=C_PROP, ms=3.2, lw=1.1, alpha=.8, label="Refer, Mond.")
    a.plot(xs, [marg[x]["sel_risk"]["mean"] for x in al], "-o", color="#334155", ms=3.2, lw=1.2, label="Risk, marg.")
    a.plot(xs, [mond[x]["sel_risk"]["mean"] for x in al], "--^", color="#334155", ms=3.2, lw=1.1, alpha=.8, label="Risk, Mond.")
    a.set_xlabel("Target error level $\\alpha$"); a.set_ylabel("Rate")
    a.set_xticks(xs); a.set_ylim(0, 0.72)
    a.set_title("(b) Referral rate and risk")
    a.legend(frameon=False, loc="upper right", fontsize=7.0, handlelength=1.3, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    save(fig, "Fig07-44.pdf")


# --------------------------------------------------------------- Fig 8
def fig08_riskcoverage(rev, o):
    rc = rev["risk_coverage"]
    cov = np.array([r["coverage"] for r in rc]); risk = np.array([r["risk"] for r in rc])
    sens = np.array([r["sens"] for r in rc]); k = np.argsort(cov)

    fig, a = plt.subplots(figsize=(3.40, 1.52))
    a.plot(cov[k], risk[k], "-o", color=C_PROP, ms=2.8, lw=1.3, label="Selective risk")
    a.plot(cov[k], 1 - sens[k], "--s", color=C_GREY, ms=2.5, lw=1.0, label="Missed abnormal")
    a.axhline(risk[0], color="k", lw=0.7, ls=":", alpha=0.6)
    a.annotate(f"forced choice {risk[0]:.3f}", xy=(0.33, risk[0] + 0.02), fontsize=7.0, alpha=0.85)
    a.set_xlabel("Coverage"); a.set_ylabel("Error on decided")
    a.set_xlim(0.3, 1.02); a.set_ylim(0, max(0.34, risk.max() + 0.06))
    a.legend(frameon=False, loc="lower right", handlelength=1.4)
    fig.tight_layout(pad=0.25)
    save(fig, "Fig08-44.pdf")


# --------------------------------------------------------------- Fig 9
def fig09_device(rev, o):
    dev = rev["device_stratified_oof"]
    dn = sorted(dev, key=lambda d: -dev[d]["n"])
    xs = np.arange(len(dn))
    sens = [dev[d]["sensitivity"] if dev[d]["sensitivity"] is not None else np.nan for d in dn]
    ms = [dev[d]["mean_score"] for d in dn]

    fig, a = plt.subplots(figsize=(3.40, 1.52))
    a.bar(xs - 0.18, sens, 0.36, color=C_ABN, label="Sensitivity")
    a.bar(xs + 0.18, ms, 0.36, color=C_GREY, label="Mean score")
    a.set_xticks(xs, [f"{d}\n{dev[d]['n_abnormal']}/{dev[d]['n_healthy']}" for d in dn],
                 fontsize=7.0)
    a.set_ylim(0, 1.36); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.legend(frameon=False, loc="upper center", ncol=2, handlelength=1.3, columnspacing=1.0)
    fig.tight_layout(pad=0.25)
    save(fig, "Fig09-44.pdf")


if __name__ == "__main__":
    rev = json.loads((RESULTS / "revision_analyses.json").read_text())
    o = np.load(RESULTS / "revision_oof.npz", allow_pickle=True)
    print("writing camera-ready figures (max 11.94 cm wide, min 7 pt text):")
    fig01_reliability()
    fig02_architecture()
    fig03_rocpr()
    fig04_confusion(rev, o)
    fig05_calibration(rev, o)
    fig06_ensemble(rev, o)
    fig07_conformal(rev, o)
    fig08_riskcoverage(rev, o)
    fig09_device(rev, o)
