from __future__ import annotations

import csv
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
AUDIO_DIR = DATA_DIR / "audio_and_txt_files"
PATIENT_CSV = DATA_DIR / "patient_diagnosis.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CACHE_FILE = RESULTS_DIR / "cycle_feature_cache.npz"
SEED = 42


@dataclass
class CycleRecord:
    patient: str
    recording: str
    device: str
    start: float
    end: float
    crackle: int
    wheeze: int
    diagnosis: str
    y_cycle: int
    y_patient: int
    features: np.ndarray


def parse_recording_name(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    patient = parts[0]
    device = parts[-1]
    return patient, device


def load_patient_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    with PATIENT_CSV.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            labels[row[0]] = row[1]
    return labels


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data)) if data.size else 1.0
    if peak > 0:
        data = data / peak
    return sr, data


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 24) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(np.array([50.0]))[0]
    high_mel = hz_to_mel(np.array([min(4000.0, sr / 2 - 1)]))[0]
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, min(center, fb.shape[1])):
            fb[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, fb.shape[1])):
            fb[i - 1, j] = (right - j) / max(right - center, 1)
    return fb


def extract_features(segment: np.ndarray, sr: int) -> np.ndarray:
    segment = segment.astype(np.float32)
    if segment.size < max(32, sr // 20):
        segment = np.pad(segment, (0, max(32, sr // 20) - segment.size))
    segment = segment - np.mean(segment)
    rms = float(np.sqrt(np.mean(segment**2) + 1e-12))
    mean_abs = float(np.mean(np.abs(segment)))
    std = float(np.std(segment))
    zcr = float(np.mean(segment[:-1] * segment[1:] < 0)) if segment.size > 1 else 0.0
    duration = float(segment.size / sr)

    n_fft = int(2 ** math.ceil(math.log2(min(max(segment.size, 512), 8192))))
    windowed = segment[:n_fft] if segment.size >= n_fft else np.pad(segment, (0, n_fft - segment.size))
    windowed = windowed * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    power = spec**2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total_power = float(np.sum(power))
    prob = power / total_power
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    cumulative = np.cumsum(power)
    rolloff = float(freqs[np.searchsorted(cumulative, 0.85 * total_power)])
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    spec_entropy = float(-np.sum(prob * np.log(prob)) / np.log(prob.size))
    dom_freq = float(freqs[int(np.argmax(power))])

    bands = [(50, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 5000)]
    band_energy = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < min(hi, sr / 2))
        band_energy.append(float(np.sum(power[mask]) / total_power) if np.any(mask) else 0.0)

    fb = mel_filterbank(sr, n_fft)
    mel_energy = fb @ power
    mfcc = dct(np.log(mel_energy + 1e-8), type=2, norm="ortho")[:13]

    return np.array(
        [
            duration,
            rms,
            mean_abs,
            std,
            zcr,
            centroid,
            bandwidth,
            rolloff,
            flatness,
            spec_entropy,
            dom_freq,
            *band_energy,
            *mfcc.tolist(),
        ],
        dtype=np.float32,
    )


def build_cycle_cache() -> dict[str, np.ndarray]:
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=True)
        return {key: data[key] for key in data.files}

    labels = load_patient_labels()
    records: list[CycleRecord] = []
    for txt_path in sorted(AUDIO_DIR.glob("*.txt")):
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        patient, device = parse_recording_name(txt_path)
        diagnosis = labels.get(patient, "Unknown")
        sr, audio = read_wav_mono(wav_path)
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            start, end = float(parts[0]), float(parts[1])
            crackle, wheeze = int(parts[2]), int(parts[3])
            s0 = max(0, int(start * sr))
            s1 = min(audio.size, int(end * sr))
            features = extract_features(audio[s0:s1], sr)
            y_cycle = int(crackle or wheeze)
            y_patient = int(diagnosis != "Healthy")
            records.append(
                CycleRecord(
                    patient=patient,
                    recording=txt_path.stem,
                    device=device,
                    start=start,
                    end=end,
                    crackle=crackle,
                    wheeze=wheeze,
                    diagnosis=diagnosis,
                    y_cycle=y_cycle,
                    y_patient=y_patient,
                    features=features,
                )
            )

    feature_matrix = np.vstack([r.features for r in records])
    cache = {
        "X": feature_matrix,
        "patient": np.array([r.patient for r in records], dtype=object),
        "recording": np.array([r.recording for r in records], dtype=object),
        "device": np.array([r.device for r in records], dtype=object),
        "diagnosis": np.array([r.diagnosis for r in records], dtype=object),
        "y_cycle": np.array([r.y_cycle for r in records], dtype=np.int8),
        "y_patient": np.array([r.y_patient for r in records], dtype=np.int8),
        "crackle": np.array([r.crackle for r in records], dtype=np.int8),
        "wheeze": np.array([r.wheeze for r in records], dtype=np.int8),
    }
    np.savez_compressed(CACHE_FILE, **cache)
    return cache


def stratified_patient_splits(patient_labels: dict[str, int]) -> dict[str, np.ndarray]:
    patients = np.array(sorted(patient_labels))
    y = np.array([patient_labels[p] for p in patients])
    train_cal, test = train_test_split(patients, test_size=0.20, random_state=SEED, stratify=y)
    y_train_cal = np.array([patient_labels[p] for p in train_cal])
    train_stage, conformal = train_test_split(
        train_cal, test_size=0.20, random_state=SEED + 1, stratify=y_train_cal
    )
    y_train_stage = np.array([patient_labels[p] for p in train_stage])
    train, calibration = train_test_split(
        train_stage, test_size=0.25, random_state=SEED + 2, stratify=y_train_stage
    )
    return {
        "train": np.array(sorted(train)),
        "calibration": np.array(sorted(calibration)),
        "conformal": np.array(sorted(conformal)),
        "test": np.array(sorted(test)),
    }


def patient_mask(patients: np.ndarray, selected: np.ndarray) -> np.ndarray:
    selected_set = set(selected.tolist())
    return np.array([p in selected_set for p in patients])


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if not np.any(mask):
            continue
        conf = np.mean(prob[mask])
        acc = np.mean(y_true[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 181), prob])
    best_score = -1.0
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        pred = (prob >= t).astype(int)
        sensitivity = recall_score(y_true, pred, zero_division=0)
        specificity = specificity_score(y_true, pred)
        score = sensitivity + specificity - 1.0
        f1 = f1_score(y_true, pred, zero_division=0)
        if score > best_score or (score == best_score and f1 > best_f1):
            best_score = score
            best_f1 = f1
            best_t = float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Specificity": specificity_score(y_true, pred),
        "ROC_AUC": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "PR_AUC": average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "ECE_lower_better": ece_score(y_true, prob),
        "Threshold": threshold,
    }
    return {k: float(v) for k, v in metrics.items()}


def sigmoid_calibrate_scores(raw_prob: np.ndarray, cal_idx: np.ndarray, y_cal: np.ndarray) -> tuple[np.ndarray, LogisticRegression]:
    clipped = np.clip(raw_prob, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    calibrator.fit(logits[cal_idx], y_cal)
    return calibrator.predict_proba(logits)[:, 1], calibrator


def aggregate_patient_features(
    patients: np.ndarray,
    devices: np.ndarray,
    y_patient_cycle: np.ndarray,
    prob: np.ndarray,
    hard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = []
    labels = []
    patient_ids = []
    for p in sorted(set(patients.tolist())):
        idx = patients == p
        p_prob = prob[idx]
        p_hard = hard[idx]
        entropy = -(
            p_prob * np.log(np.clip(p_prob, 1e-8, 1.0))
            + (1 - p_prob) * np.log(np.clip(1 - p_prob, 1e-8, 1.0))
        ) / np.log(2)
        reliability = 1.0 - entropy
        weighted = float(np.sum(reliability * p_prob) / max(np.sum(reliability), 1e-8))
        row = [
            float(np.mean(p_hard)),
            float(np.mean(p_prob)),
            weighted,
            float(np.std(p_prob)),
            float(np.min(p_prob)),
            float(np.percentile(p_prob, 25)),
            float(np.median(p_prob)),
            float(np.percentile(p_prob, 75)),
            float(np.percentile(p_prob, 90)),
            float(np.max(p_prob)),
            float(np.mean(entropy)),
            float(np.sum(p_prob > 0.5)),
            float(np.log1p(np.sum(idx))),
            float(len(set(devices[idx].tolist()))),
        ]
        rows.append(row)
        labels.append(int(y_patient_cycle[idx][0]))
        patient_ids.append(p)
    names = [
        "count_ratio",
        "mean_prob",
        "confidence_weighted_fraction",
        "std_prob",
        "min_prob",
        "p25_prob",
        "median_prob",
        "p75_prob",
        "p90_prob",
        "max_prob",
        "mean_entropy",
        "n_high_prob",
        "log_cycle_count",
        "device_count",
    ]
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=int), np.array(patient_ids), names


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def conformal_curve(y_cal: np.ndarray, p_cal: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> list[dict[str, float]]:
    out = []
    nonconformity = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n = len(nonconformity)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        q_idx = min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1)
        q = float(np.sort(nonconformity)[q_idx])
        include_neg = p_test <= q
        include_pos = (1.0 - p_test) <= q
        covered = np.where(y_test == 1, include_pos, include_neg)
        single = include_neg ^ include_pos
        wrong_single = single & (((p_test >= 0.5).astype(int)) != y_test)
        out.append(
            {
                "alpha": alpha,
                "qhat": q,
                "coverage": float(np.mean(covered)),
                "single_error": float(np.mean(wrong_single)),
                "referral_rate": float(np.mean(include_neg & include_pos)),
                "empty_rate": float(np.mean(~include_neg & ~include_pos)),
            }
        )
    return out


def plot_reliability(path: Path, y_true: np.ndarray, before: np.ndarray, after: np.ndarray) -> None:
    plt.figure(figsize=(5.0, 3.6))
    bins = np.linspace(0, 1, 8)
    for prob, label in [(before, "uncalibrated"), (after, "calibrated")]:
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
            if np.any(mask):
                xs.append(float(np.mean(prob[mask])))
                ys.append(float(np.mean(y_true[mask])))
        plt.plot(xs, ys, marker="o", label=f"{label} ECE={ece_score(y_true, prob):.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical abnormal-event rate")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_roc_pr(path: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(y, p):.2f}")
        axes[1].plot(rec, prec, label=f"{label} AP={average_precision_score(y, p):.2f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], ylabel: str) -> None:
    plt.figure(figsize=(5.5, 3.2))
    plt.bar(labels, values, color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"][: len(labels)])
    plt.ylabel(ylabel)
    plt.xticks(rotation=18, ha="right")
    plt.ylim(0, max(1.0, max(values) * 1.15))
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_conformal(path: Path, rows: list[dict[str, float]]) -> None:
    alphas = [r["alpha"] for r in rows]
    coverage = [r["coverage"] for r in rows]
    referral = [r["referral_rate"] for r in rows]
    target = [1 - a for a in alphas]
    plt.figure(figsize=(5.4, 3.4))
    plt.plot(alphas, coverage, marker="o", label="empirical coverage")
    plt.plot(alphas, target, "--", label="target coverage")
    plt.plot(alphas, referral, marker="s", label="refer rate")
    plt.xlabel("alpha")
    plt.ylabel("fraction")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    cache = build_cycle_cache()
    X = cache["X"]
    patients = cache["patient"]
    devices = cache["device"]
    diagnoses = cache["diagnosis"]
    y_cycle = cache["y_cycle"].astype(int)
    y_patient_cycle = cache["y_patient"].astype(int)
    patient_labels = {p: int(y_patient_cycle[patients == p][0]) for p in sorted(set(patients.tolist()))}
    splits = stratified_patient_splits(patient_labels)

    masks = {name: patient_mask(patients, split) for name, split in splits.items()}
    split_summary = {
        name: {
            "patients": int(len(split)),
            "abnormal_patients": int(sum(patient_labels[p] for p in split)),
            "healthy_patients": int(len(split) - sum(patient_labels[p] for p in split)),
        }
        for name, split in splits.items()
    }

    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=220, max_depth=12, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_pipe = make_pipeline(scaler, rf)
    rf_pipe.fit(X[masks["train"]], y_cycle[masks["train"]])
    uncal_test = rf_pipe.predict_proba(X[masks["test"]])[:, 1]
    calibrator = CalibratedClassifierCV(FrozenEstimator(rf_pipe), method="isotonic")
    calibrator.fit(X[masks["calibration"]], y_cycle[masks["calibration"]])
    all_prob = calibrator.predict_proba(X)[:, 1]
    all_hard = (all_prob >= 0.5).astype(int)
    cal_test = all_prob[masks["test"]]

    cycle_threshold = best_threshold(y_cycle[masks["conformal"]], all_prob[masks["conformal"]])
    cycle_metrics = binary_metrics(y_cycle[masks["test"]], cal_test, cycle_threshold)
    cycle_metrics["Uncalibrated_ECE"] = ece_score(y_cycle[masks["test"]], uncal_test)

    P, Py, Pids, feature_names = aggregate_patient_features(patients, devices, y_patient_cycle, all_prob, all_hard)
    patient_index = {p: i for i, p in enumerate(Pids)}
    psplit_idx = {name: np.array([patient_index[p] for p in split]) for name, split in splits.items()}

    train_agg = psplit_idx["train"]
    calibration_idx = psplit_idx["calibration"]
    conformal_idx = psplit_idx["conformal"]
    test_idx = psplit_idx["test"]

    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    models["Count-ratio (BME106-style)"] = (P[:, 0], P[:, 0], P[:, 0])
    models["Confidence-weighted fraction"] = (P[:, 2], P[:, 2], P[:, 2])

    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
    proposed_name = "CARE-Lung (regularized distribution aggregator)"
    lr.fit(P[train_agg], Py[train_agg])
    lr_prob = lr.predict_proba(P)[:, 1]
    models[proposed_name] = (lr_prob, lr_prob, lr_prob)

    gb = GradientBoostingClassifier(random_state=SEED, n_estimators=80, max_depth=2, learning_rate=0.06)
    gb.fit(P[train_agg], Py[train_agg])
    gb_raw_prob = gb.predict_proba(P)[:, 1]
    gb_prob, gb_sigmoid = sigmoid_calibrate_scores(gb_raw_prob, calibration_idx, Py[calibration_idx])
    models["Gradient-boosted distribution aggregator"] = (gb_prob, gb_prob, gb_prob)

    svm_patient = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED),
    )
    svm_patient.fit(P[train_agg], Py[train_agg])
    svm_prob = svm_patient.predict_proba(P)[:, 1]
    models["SVM + spectral summaries"] = (svm_prob, svm_prob, svm_prob)

    rows = []
    curves = {}
    thresholds = {}
    for name, (score_all, _, _) in models.items():
        threshold = best_threshold(Py[conformal_idx], score_all[conformal_idx])
        thresholds[name] = threshold
        met = binary_metrics(Py[test_idx], score_all[test_idx], threshold)
        rows.append({"Model": name, **{k: round(v, 4) for k, v in met.items()}})
        curves[name] = (Py[test_idx], score_all[test_idx])

    write_csv(
        TABLE_DIR / "main_comparison.csv",
        ["Model", "Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC_AUC", "PR_AUC", "ECE_lower_better", "Threshold"],
        rows,
    )

    care_prob = lr_prob
    care_threshold = thresholds[proposed_name]
    care_pred = (care_prob[test_idx] >= care_threshold).astype(int)
    cm = confusion_matrix(Py[test_idx], care_pred, labels=[0, 1])
    confusion_rows = [
        {"": "True Negative (healthy)", "Pred Negative": int(cm[0, 0]), "Pred Positive": int(cm[0, 1])},
        {"": "True Positive (abnormal)", "Pred Negative": int(cm[1, 0]), "Pred Positive": int(cm[1, 1])},
    ]
    write_csv(TABLE_DIR / "confusion_matrix.csv", ["", "Pred Negative", "Pred Positive"], confusion_rows)

    ablation_rows = []
    for name in [
        "Count-ratio (BME106-style)",
        "Confidence-weighted fraction",
        proposed_name,
        "Gradient-boosted distribution aggregator",
    ]:
        score = models[name][0]
        threshold = thresholds[name]
        met = binary_metrics(Py[test_idx], score[test_idx], threshold)
        ablation_rows.append(
            {
                "Aggregation": name,
                "Patient_AUROC": round(met["ROC_AUC"], 4),
                "F1": round(met["F1"], 4),
                "Sensitivity": round(met["Recall"], 4),
                "Specificity": round(met["Specificity"], 4),
            }
        )
    write_csv(TABLE_DIR / "aggregation_ablation.csv", list(ablation_rows[0].keys()), ablation_rows)

    patient_device = []
    for p in Pids:
        ds = devices[patients == p]
        patient_device.append(Counter(ds.tolist()).most_common(1)[0][0])
    patient_device = np.array(patient_device)
    device_rows = []
    for dev in sorted(set(patient_device[test_idx].tolist())):
        idx = test_idx[patient_device[test_idx] == dev]
        if len(idx) == 0:
            continue
        prob = care_prob[idx]
        y = Py[idx]
        pred = (prob >= care_threshold).astype(int)
        has_positive = np.any(y == 1)
        has_negative = np.any(y == 0)
        device_rows.append(
            {
                "Stethoscope": dev,
                "N_patients": int(len(idx)),
                "N_abnormal": int(np.sum(y)),
                "AUROC": round(roc_auc_score(y, prob), 4) if len(np.unique(y)) > 1 else "NA",
                "Sensitivity": round(recall_score(y, pred, zero_division=0), 4) if has_positive else "NA",
                "Specificity": round(specificity_score(y, pred), 4) if has_negative else "NA",
            }
        )
    write_csv(TABLE_DIR / "per_device.csv", ["Stethoscope", "N_patients", "N_abnormal", "AUROC", "Sensitivity", "Specificity"], device_rows)

    conformal_rows = conformal_curve(Py[conformal_idx], care_prob[conformal_idx], Py[test_idx], care_prob[test_idx])
    write_csv(TABLE_DIR / "conformal.csv", list(conformal_rows[0].keys()), [{k: round(v, 4) for k, v in r.items()} for r in conformal_rows])

    model_names_list = list(models.keys())
    eval_artifacts = {
        "model_names": np.array(model_names_list, dtype=object),
        "score_matrix_test": np.array([models[n][0][test_idx] for n in model_names_list], dtype=np.float64),
        "thresholds": np.array([thresholds[n] for n in model_names_list], dtype=np.float64),
        "test_patient_ids": Pids[test_idx],
        "y_test_patient": Py[test_idx],
        "test_patient_device": patient_device[test_idx],
        "y_cycle_test": y_cycle[masks["test"]],
        "uncal_cycle_prob_test": uncal_test,
        "cal_cycle_prob_test": cal_test,
        "care_prob_conformal": care_prob[conformal_idx],
        "y_conformal_patient": Py[conformal_idx],
        "care_threshold": np.array([care_threshold], dtype=np.float64),
        "feature_names": np.array(feature_names, dtype=object),
        "patient_feature_matrix_test": P[test_idx],
    }
    np.savez_compressed(RESULTS_DIR / "eval_artifacts.npz", **eval_artifacts)

    device_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    device_clf.fit(X[masks["train"]], devices[masks["train"]])
    device_acc = float(device_clf.score(X[masks["test"]], devices[masks["test"]]))
    majority_device_acc = float(Counter(devices[masks["test"]].tolist()).most_common(1)[0][1] / np.sum(masks["test"]))

    start = time.perf_counter()
    _ = calibrator.predict_proba(X[masks["test"]][: min(1000, np.sum(masks["test"]))])
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / min(1000, np.sum(masks["test"])) * 1000
    model_path = RESULTS_DIR / "care_lung_models.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"cycle": calibrator, "aggregator": gb, "aggregator_sigmoid": gb_sigmoid, "feature_names": feature_names}, f)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    plot_reliability(FIG_DIR / "reliability.png", y_cycle[masks["test"]], uncal_test, cal_test)
    plot_roc_pr(
        FIG_DIR / "patient_roc_pr.png",
        {
            "count-ratio": curves["Count-ratio (BME106-style)"],
            "weighted": curves["Confidence-weighted fraction"],
            "CARE-Lung": curves[proposed_name],
        },
    )
    plot_bar(
        FIG_DIR / "ablation_f1.png",
        [r["Aggregation"].replace(" aggregator", "").replace(" (BME106-style)", "") for r in ablation_rows],
        [float(r["F1"]) for r in ablation_rows],
        "Patient F1",
    )
    plot_conformal(FIG_DIR / "conformal.png", conformal_rows)

    summary = {
        "dataset": {
            "cycles": int(len(y_cycle)),
            "recordings": int(len(set(cache["recording"].tolist()))),
            "patients": int(len(patient_labels)),
            "diagnosis_counts": dict(Counter(load_patient_labels().values())),
            "cycle_label_counts": {"normal": int(np.sum(y_cycle == 0)), "abnormal_event": int(np.sum(y_cycle == 1))},
            "device_counts_cycles": dict(Counter(devices.tolist())),
        },
        "splits": split_summary,
        "cycle_metrics": {k: round(float(v), 4) for k, v in cycle_metrics.items()},
        "device_leakage_probe": {
            "cycle_feature_device_accuracy": round(device_acc, 4),
            "majority_baseline": round(majority_device_acc, 4),
        },
        "edge_proxy": {
            "desktop_latency_ms_per_cycle": round(float(latency_ms), 4),
            "serialized_model_size_mb": round(float(model_size_mb), 4),
        },
        "best_model_by_f1": max(rows, key=lambda r: r["F1"]),
    }
    (RESULTS_DIR / "study_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
