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
    # savefig.bbox="tight" silently applies the matplotlib default of
    # 0.1in padding on every side (0.2in added to width and height) unless
    # pad_inches is overridden here -- that invisible margin is what pushed
    # Fig 2 to 12.21 cm despite its visible ink measuring only 11.51 cm.
    fig.savefig(p, pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------- Fig 1
def fig01_reliability():
    art = np.load(RESULTS / "eval_artifacts.npz", allow_pickle=True)
    y = art["y_cycle_test"].astype(int)
    before, after = art["uncal_cycle_prob_test"], art["cal_cycle_prob_test"]

    # both panels use the same two series, so one shared legend below the
    # figure keeps it off the curves and the histogram bars
    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.62))
    bins = np.linspace(0, 1, 9)
    handles, labels = [], []
    for prob, lab, col, mk in [(before, "Uncalibrated", C_GREY, "o"),
                               (after, "Isotonic-calibrated", C_PROP, "s")]:
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
            if np.any(m):
                xs.append(float(np.mean(prob[m]))); ys.append(float(np.mean(y[m])))
                ns.append(int(m.sum()))
        ece = float(np.sum(np.array(ns) / len(prob) * np.abs(np.array(ys) - np.array(xs))))
        line, = ax[0].plot(xs, ys, marker=mk, ms=3.4, lw=1.4, color=col)
        handles.append(line); labels.append(f"{lab} ({ece:.3f})")
    ref, = ax[0].plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6)
    handles.append(ref); labels.append("Perfect")
    ax[0].set_xlabel("Mean predicted probability")
    ax[0].set_ylabel("Empirical abnormal rate")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.04)
    ax[0].set_title("(a) Reliability curve")

    ax[1].hist(before, bins=bins, alpha=0.55, color=C_GREY, density=True)
    ax[1].hist(after, bins=bins, alpha=0.55, color=C_PROP, density=True)
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Density")
    ax[1].set_title("(b) Score distribution")

    fig.tight_layout(pad=0.3, rect=(0, 0.13, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.0, handlelength=1.4, columnspacing=1.0,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.045))
    save(fig, "Fig01-44.pdf")


# --------------------------------------------------------------- Fig 2
def fig02_architecture():
    """Pipeline diagram, same content and layout as the original design
    (full descriptive labels, five nodes in row (a), the conformal set
    notation spelled out in row (c)), but built directly at print size.

    The original was designed on an 18.2 x 11.6 cm canvas at 6.1-6.7 pt and
    then placed at 0.90\\textwidth in the paper, which scaled it down by
    0.605x -- so its labels actually printed at 3.7-4.1 pt, well under the
    7 pt Springer floor. That is the defect the terse rewrite was trying to
    fix, but it over-corrected by cutting content instead of just building
    the same design at true size. Here the coordinate system is inches, so
    box sizes below can be read directly as physical dimensions, and no
    `\\includegraphics` width= scaling is ever applied to this file.
    """
    W, H = 4.70, 3.55  # inches; 11.94 x 9.02 cm, under the 12 cm cap
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    _bounds_violations = []

    def box(cx, cy, w, h, text, fc, fontsize=7.1, fontweight="normal", ec="#374151"):
        left, right = cx - w / 2, cx + w / 2
        bottom, top = cy - h / 2, cy + h / 2
        if left < 0 or right > W or bottom < 0 or top > H:
            _bounds_violations.append(
                f"{text.splitlines()[0]!r}: [{left:.2f},{right:.2f}]x[{bottom:.2f},{top:.2f}] "
                f"outside canvas [0,{W}]x[0,{H}]")
        ax.add_patch(FancyBboxPatch(
            (left, bottom), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=0.9, edgecolor=ec, facecolor=fc, zorder=2,
        ))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color="#111827", zorder=3, linespacing=1.30)
        return (cx, cy, w, h)

    def link(b1, b2, p1=None, p2=None, rad=0.0, color="#4b5563", lw=0.9):
        x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
        P = {"r": (x1 + w1 / 2, y1), "l": (x1 - w1 / 2, y1),
             "t": (x1, y1 + h1 / 2), "b": (x1, y1 - h1 / 2)}
        Q = {"r": (x2 + w2 / 2, y2), "l": (x2 - w2 / 2, y2),
             "t": (x2, y2 + h2 / 2), "b": (x2, y2 - h2 / 2)}
        start = P[p1] if p1 else (x1, y1)
        end = Q[p2] if p2 else (x2, y2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6.5,
                                     lw=lw, color=color, zorder=1, shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f"arc3,rad={rad}"))

    def stage_label(x, y, text, color):
        ax.text(x, y, text, color=color, va="top", ha="left", fontsize=7.6,
                fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    # Three row-bands of equal height, each with its own label strip above it.
    band_h = H / 3
    y1 = H - band_h / 2 - 0.14   # row (a) center
    y2 = H - band_h - band_h / 2 - 0.10   # row (b) center
    y3 = band_h / 2 - 0.04                 # row (c) center

    # === Row (a): per-cycle calibration, 5 nodes left to right ==============
    bw_a, gap_a = 0.86, 0.045
    xs_a = [0.05 + bw_a / 2 + i * (bw_a + gap_a) for i in range(5)]
    a1 = box(xs_a[0], y1, bw_a, 0.72,
             "Respiratory\ncycle\n(ICBHI)", "#eef2ff", fontsize=7.0)
    a2 = box(xs_a[1], y1, bw_a, 0.72,
             "Acoustic\nfeatures\n(31-d, MFCC)", "#e0e7ff", fontsize=7.0)
    a3 = box(xs_a[2], y1, bw_a, 0.72,
             "Random forest\n(RF-train,\nbalanced)", "#dbeafe", fontsize=7.0)
    a4 = box(xs_a[3], y1, bw_a, 0.72,
             "Isotonic\ncalibration\n(calibration)", "#cffafe", fontsize=7.0)
    a5 = box(xs_a[4], y1, bw_a, 0.72,
             "Calibrated\nposterior\nECE 0.198\nto 0.147", "#fef9c3",
             fontweight="bold", fontsize=7.0)
    for p, q in [(a1, a2), (a2, a3), (a3, a4), (a4, a5)]:
        link(p, q, "r", "l")

    # === Row (b): cycle-to-patient aggregation, right to left ===============
    b1 = box(xs_a[4], y2, bw_a, 0.72,
             "Cycle to\npatient\n(14-d summary)",
             "#fde68a", fontsize=7.0)
    b2 = box(2.35, y2, 1.30, 0.72,
             "CARE-Lung aggregator\n(regularised logistic\nregression)",
             "#fecaca", fontweight="bold", fontsize=7.0)
    b3 = box(0.75, y2, 1.30, 0.72,
             "Patient screening score\nAUROC 0.832, F1 0.809",
             "#fee2e2", fontweight="bold", fontsize=7.0)
    link(a5, b1, "b", "t")
    link(b1, b2, "l", "r"); link(b2, b3, "l", "r")

    # === Row (c): calibrated decision and conformal triage ==================
    c1 = box(1.00, y3, 1.85, 0.72,
             "Operating-point decision:\nYouden's J on the held-out\n"
             "threshold-selection split\n-> screen-/screen+",
             "#dcfce7", fontsize=7.0)
    c2 = box(3.25, y3, 2.55, 0.72,
             "Split-conformal referral: nonconformity\n"
             "1 - score (abnormal), score (healthy);\n"
             "quantile at target error alpha ->\n"
             "{healthy}, {abnormal}, or both = REFER",
             "#bbf7d0", fontweight="bold", fontsize=7.0)
    link(b3, c1, "b", "t", rad=0.22)
    link(b3, c2, "b", "t", rad=-0.14)

    # band boundaries, top to bottom: H -> 2*band_h -> band_h -> 0
    stage_label(0.03, H - 0.06, "(a)  Per-cycle feature extraction and calibration", "#1e3a8a")
    stage_label(0.03, 2 * band_h - 0.02, "(b)  Cycle-to-patient aggregation", "#92400e")
    stage_label(0.03, band_h - 0.02, "(c)  Calibrated decision and conformal triage", "#065f46")

    for yy in (2 * band_h, band_h):
        ax.plot([0.03, W - 0.03], [yy, yy], color="#d1d5db", lw=0.6,
                ls=(0, (4, 3)), zorder=0)

    if _bounds_violations:
        raise RuntimeError("Fig 2 boxes exceed the canvas:\n  " + "\n  ".join(_bounds_violations))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "Fig02-44.pdf")


# --------------------------------------------------------------- Fig 3
def fig03_rocpr():
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    art = np.load(RESULTS / "cv_eval_artifacts.npz", allow_pickle=True)
    names = [str(n) for n in art["model_names"]]
    y = art["y_true"].astype(int)
    prob = {n: art[f"oof_prob__{i}"] for i, n in enumerate(names)}

    # six series will not fit inside either panel without covering the curves,
    # so the legend is placed under both panels as a shared figure legend
    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.78))
    handles, labels = [], []
    for n in SHORT:
        fpr, tpr, _ = roc_curve(y, prob[n])
        lw = 1.9 if n == PROP else 1.1
        line, = ax[0].plot(fpr, tpr, color=COLORS[n], lw=lw,
                           zorder=5 if n == PROP else 2)
        handles.append(line)
        labels.append(f"{SHORT[n]} ({roc_auc_score(y, prob[n]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.45)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) Pooled out-of-fold ROC")
    ax[0].set_xlim(-0.02, 1.02); ax[0].set_ylim(-0.02, 1.04)

    prev = float(np.mean(y))
    for n in SHORT:
        pr, rc, _ = precision_recall_curve(y, prob[n])
        lw = 1.9 if n == PROP else 1.1
        ax[1].plot(rc, pr, color=COLORS[n], lw=lw, zorder=5 if n == PROP else 2)
    ax[1].axhline(prev, color="k", ls="--", lw=0.8, alpha=0.45)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("(b) Precision-recall")
    ax[1].set_xlim(-0.02, 1.02); ax[1].set_ylim(0.4, 1.04)
    fig.tight_layout(pad=0.3, rect=(0, 0.19, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.0, handlelength=1.5, columnspacing=1.4,
               handletextpad=0.5, labelspacing=0.35,
               bbox_to_anchor=(0.5, -0.045))
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
    # short label: the panel is only ~4 cm tall, so the full phrase would clip
    a.set_ylabel("ECE")
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

    # Explicit geometry rather than tight_layout: the two legends live below
    # their own panels, and tight_layout does not reserve room for artists
    # anchored outside the axes, which previously squashed the plots.
    fig, ax = plt.subplots(1, 2, figsize=(MAXW, 1.92))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.55, wspace=0.40)
    a = ax[0]
    a.plot(xs, 1 - xs, "k:", lw=0.9, label="Target $1-\\alpha$")
    a.plot(xs, [marg[x]["cov_abn"]["mean"] for x in al], "-o", color=C_ABN, ms=3.2, lw=1.2, label="Marg., abn.")
    a.plot(xs, [marg[x]["cov_hea"]["mean"] for x in al], "-o", color=C_HEA, ms=3.2, lw=1.2, label="Marg., healthy")
    a.plot(xs, [mond[x]["cov_abn"]["mean"] for x in al], "--^", color=C_ABN, ms=3.2, lw=1.1, alpha=.85, label="Mond., abn.")
    a.plot(xs, [mond[x]["cov_hea"]["mean"] for x in al], "--^", color=C_HEA, ms=3.2, lw=1.1, alpha=.85, label="Mond., healthy")
    a.set_xlabel("Target error level $\\alpha$"); a.set_ylabel("Empirical coverage")
    a.set_xticks(xs); a.set_ylim(0.60, 1.05)
    a.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_title("(a) Class-conditional coverage")
    ha, la = a.get_legend_handles_labels()

    a = ax[1]
    a.plot(xs, [marg[x]["refer"]["mean"] for x in al], "-o", color=C_PROP, ms=3.2, lw=1.2, label="Refer, marg.")
    a.plot(xs, [mond[x]["refer"]["mean"] for x in al], "--^", color=C_PROP, ms=3.2, lw=1.1, alpha=.8, label="Refer, Mond.")
    a.plot(xs, [marg[x]["sel_risk"]["mean"] for x in al], "-o", color="#334155", ms=3.2, lw=1.2, label="Risk, marg.")
    a.plot(xs, [mond[x]["sel_risk"]["mean"] for x in al], "--^", color="#334155", ms=3.2, lw=1.1, alpha=.8, label="Risk, Mond.")
    a.set_xlabel("Target error level $\\alpha$"); a.set_ylabel("Rate")
    a.set_xticks(xs); a.set_ylim(0, 0.62)
    a.set_yticks([0, 0.2, 0.4, 0.6])
    a.set_title("(b) Referral rate and risk")
    hb, lb = a.get_legend_handles_labels()

    # legends anchored in figure coordinates, centred under each panel
    lkw = dict(frameon=False, fontsize=7.0, ncol=2, handlelength=1.3,
               labelspacing=0.32, columnspacing=1.0, handletextpad=0.4,
               borderpad=0.0, loc="upper center")
    fig.legend(ha, la, bbox_to_anchor=(0.30, 0.295), **lkw)
    fig.legend(hb, lb, bbox_to_anchor=(0.79, 0.295), **lkw)
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
