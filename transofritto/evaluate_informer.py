# evaluator.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Optional dependencies (recommended)
from scipy.stats import spearmanr, pearsonr, ks_2samp, wasserstein_distance

from sklearn.metrics import r2_score, confusion_matrix

import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray]


def _safe_normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure p is a valid probability distribution along last axis.
    Handles:
      - raw probabilities (may not sum to 1)
      - logits (if negative values present and rows don't resemble probs)
      - log-probabilities (if many negatives and exp sums ~ 1)
    Strategy:
      1) If any NaNs -> raise
      2) If values are all >=0 and row sums are close to 1 -> treat as probs
      3) Else if values are mostly <=0 and exp(row) sums close to 1 -> treat as log-probs
      4) Else -> treat as logits -> softmax
    """
    if np.isnan(p).any():
        raise ValueError("Input contains NaNs.")

    p = np.asarray(p, dtype=np.float64)

    # If already nonnegative and sums approximately 1, treat as probs
    row_sum = p.sum(axis=-1, keepdims=True)
    if (p >= -1e-12).all() and np.allclose(row_sum, 1.0, atol=1e-3):
        p = np.clip(p, eps, 1.0)
        p = p / p.sum(axis=-1, keepdims=True)
        return p

    # Check if looks like log-probs: exp sums ~ 1
    exp_sum = np.exp(np.clip(p, -80, 80)).sum(axis=-1, keepdims=True)
    if np.allclose(exp_sum, 1.0, atol=1e-3):
        probs = np.exp(np.clip(p, -80, 80))
        probs = np.clip(probs, eps, 1.0)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs

    # Otherwise treat as logits: softmax
    x = p - p.max(axis=-1, keepdims=True)
    probs = np.exp(np.clip(x, -80, 80))
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def _kl_divergence_batch(true_p: np.ndarray, pred_p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    KL(Q || P) per sample, where Q=true, P=pred.
    Both inputs must be valid probabilities (sum to 1).
    Returns shape: (N,)
    """
    Q = np.clip(true_p, eps, 1.0)
    P = np.clip(pred_p, eps, 1.0)
    return np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)


def _cdf_ks_discrete(true_p: np.ndarray, pred_p: np.ndarray) -> np.ndarray:
    """
    Discrete KS statistic between two categorical distributions:
      KS = max_i |CDF_true(i) - CDF_pred(i)|
    Returns shape: (N,)
    """
    cdf_t = np.cumsum(true_p, axis=-1)
    cdf_p = np.cumsum(pred_p, axis=-1)
    return np.max(np.abs(cdf_t - cdf_p), axis=-1)


def _wasserstein_1d_discrete(true_p: np.ndarray, pred_p: np.ndarray) -> np.ndarray:
    """
    1D Wasserstein distance between two discrete distributions on support {0..C-1}.
    Returns shape: (N,)
    """
    n, c = true_p.shape
    support = np.arange(c, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = wasserstein_distance(support, support, u_weights=true_p[i], v_weights=pred_p[i])
    return out


def _vector_metrics(true_p: np.ndarray, pred_p: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    """
    Per-sample metrics computed over the 16-d vector.
    Returns dict of arrays, each shape (N,)
    """
    diff = pred_p - true_p
    abs_diff = np.abs(diff)

    mse = np.mean(diff ** 2, axis=-1)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff, axis=-1)

    # MAPE (percentage), defined elementwise as |(y - yhat)/y|, averaged over dims
    # Note: probabilities can include zeros, so we stabilize with eps.
    mape = np.mean(abs_diff / np.clip(np.abs(true_p), eps, None), axis=-1) * 100.0

    # ARFE: Average Relative Fraction Error (commonly used similarly to relative error on fractions)
    # Here: mean_i |pred_i - true_i| / (true_i + eps)
    arfe = np.mean(abs_diff / np.clip(true_p, eps, None), axis=-1)

    # Spearman and Pearson per sample:
    # Compute correlation between the 16 fractions of a single bin.
    # Spearman can be undefined if constant; handle by returning NaN and later aggregating safely.
    spearman = np.empty(true_p.shape[0], dtype=np.float64)
    pearson = np.empty(true_p.shape[0], dtype=np.float64)
    for i in range(true_p.shape[0]):
        # Spearman
        rho, _ = spearmanr(true_p[i], pred_p[i])
        spearman[i] = rho

        # Pearson
        r, _ = pearsonr(true_p[i], pred_p[i])
        pearson[i] = r

    # R^2 per sample across the 16 dims (treat each bin as a 16-d regression target)
    # sklearn's r2_score supports multioutput; we do per-sample manually.
    r2 = np.empty(true_p.shape[0], dtype=np.float64)
    for i in range(true_p.shape[0]):
        r2[i] = r2_score(true_p[i], pred_p[i])

    # KS and Wasserstein on discrete distributions
    ks = _cdf_ks_discrete(true_p, pred_p)
    wdist = _wasserstein_1d_discrete(true_p, pred_p)

    # Argmax fraction error (absolute difference in peak fraction index)
    true_arg = np.argmax(true_p, axis=-1)
    pred_arg = np.argmax(pred_p, axis=-1)
    argmax_err = np.abs(pred_arg - true_arg).astype(np.float64)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "arfe": arfe,
        "spearman": spearman,
        "pearson": pearson,
        "r2": r2,
        "ks": ks,
        "wasserstein": wdist,
        "argmax_err": argmax_err,
        "true_argmax": true_arg.astype(np.int64),
        "pred_argmax": pred_arg.astype(np.int64),
    }


def _nan_safe_agg(x: np.ndarray) -> Dict[str, float]:
    """Return mean/median/std ignoring NaNs."""
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.nanmean(x)),
        "median": float(np.nanmedian(x)),
        "std": float(np.nanstd(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }


@dataclass
class EvaluationOutputs:
    summary: Dict[str, Dict[str, float]]
    per_bin: Dict[str, np.ndarray]
    confusion: np.ndarray
    paths: Dict[str, str]


class ReplicationTimingEvaluator:
    """
    Evaluates predicted vs true 16-fraction RT distributions.

    Expected shapes:
      true: (N, C) where C=16 (probabilities)
      pred: (N, C) probabilities OR logits OR log-probabilities

    Outputs:
      - summary_metrics.json
      - per_bin_metrics.csv
      - confusion_matrix.csv
      - confusion_matrix.png
    """

    def __init__(
            self,
            out_dir: str = "eval_outputs",
            class_names: Optional[list] = None,
            eps: float = 1e-12,
    ):
        self.out_dir = out_dir
        self.eps = eps
        self.class_names = class_names or [str(i) for i in range(16)]
        os.makedirs(self.out_dir, exist_ok=True)

    def load_from_npy(self, pred_path: str, true_path: str) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.load(pred_path)
        true = np.load(true_path)
        return pred, true

    def evaluate(
            self,
            pred: ArrayLike,
            true: ArrayLike,
            prefix: str = "test",
            save_per_bin_csv: bool = True,
    ) -> EvaluationOutputs:
        pred = np.asarray(pred)
        true = np.asarray(true)


        # Accept (N, C) or (N, 1, C) or (N, T, C)
        if pred.ndim == 3:
            # If T=1, squeeze. If T>1, flatten N*T samples (consistent for per-bin metrics)
            if pred.shape[1] == 1:
                pred = pred[:, 0, :]
                true = true[:, 0, :]
            else:
                n, t, c = pred.shape
                pred = pred.reshape(n * t, c)
                true = true.reshape(n * t, c)

        if pred.ndim != 2 or true.ndim != 2:
            raise ValueError(f"pred and true must be 2D arrays. Got pred{pred.shape}, true{true.shape}.")

        # Normalize / convert representations
        true_p = _safe_normalize_probs(true, eps=self.eps)
        pred_p = _safe_normalize_probs(pred, eps=self.eps)

        # Primary loss: KL(Q || P) per bin
        kl = _kl_divergence_batch(true_p, pred_p, eps=self.eps)

        # Other metrics (per-bin)
        per_bin = _vector_metrics(true_p, pred_p, eps=self.eps)
        per_bin["kl"] = kl

        # Confusion matrix on argmax fraction
        y_true = per_bin["true_argmax"]
        y_pred = per_bin["pred_argmax"]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))

        # Summary aggregation
        summary = {}
        for k, v in per_bin.items():
            if k in ("true_argmax", "pred_argmax"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 1:
                summary[k] = _nan_safe_agg(v)

        # Save artifacts
        paths = {}
        paths["summary_json"] = os.path.join(self.out_dir, f"{prefix}_summary_metrics.json")
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Confusion CSV
        paths["confusion_csv"] = os.path.join(self.out_dir, f"{prefix}_confusion_matrix.csv")
        np.savetxt(paths["confusion_csv"], cm, delimiter=",", fmt="%d")

        # Confusion plot
        paths["confusion_png"] = os.path.join(self.out_dir, f"{prefix}_confusion_matrix.png")
        self._save_confusion_plot(cm, paths["confusion_png"], title=f"Confusion Matrix ({prefix})")

        # Per-bin CSV
        if save_per_bin_csv:
            paths["per_bin_csv"] = os.path.join(self.out_dir, f"{prefix}_per_bin_metrics.csv")
            self._save_per_bin_csv(per_bin, paths["per_bin_csv"])

        return EvaluationOutputs(summary=summary, per_bin=per_bin, confusion=cm, paths=paths)

    def _save_per_bin_csv(self, per_bin: Dict[str, np.ndarray], out_path: str) -> None:
        # Flatten into columns
        keys = [k for k in per_bin.keys() if isinstance(per_bin[k], np.ndarray)]
        n = None
        for k in keys:
            if per_bin[k].ndim == 1:
                n = per_bin[k].shape[0]
                break
        if n is None:
            raise ValueError("No 1D per-bin arrays found to write.")

        cols = []
        header = []
        for k in keys:
            v = per_bin[k]
            if v.ndim == 1:
                cols.append(v.reshape(n, 1))
                header.append(k)

        mat = np.concatenate(cols, axis=1)
        header_line = ",".join(header)
        np.savetxt(out_path, mat, delimiter=",", header=header_line, comments="")

    def _save_confusion_plot(self, cm: np.ndarray, out_path: str, title: str) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Inverted grayscale
        im = ax.imshow(cm, aspect="auto", cmap="gray_r")

        ax.set_title(title)
        ax.set_xlabel("Predicted argmax fraction")
        ax.set_ylabel("True argmax fraction")

        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        # Annotate counts
        # For readability on inverted grayscale, switch text color based on intensity
        vmax = cm.max() if cm.size else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                text_color = "black" if val > (0.5 * vmax) else "white"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    # Example CLI-like run (edit paths as needed)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pred_path = str(PROJECT_ROOT / "transoffritto/transofritto/results/mNPC_val6_test9/pred.npy")
    true_path = str(PROJECT_ROOT / "transoffritto/transofritto/results/mNPC_val6_test9/true.npy")

    evaluator = ReplicationTimingEvaluator(out_dir="eval_outputs", class_names=[str(i) for i in range(16)])
    pred, true = evaluator.load_from_npy(pred_path, true_path)

    outputs = evaluator.evaluate(pred, true, prefix="chrom_test", save_per_bin_csv=True)

    print("Saved:")
    for k, v in outputs.paths.items():
        print(f"  {k}: {v}")

    print("\nKey summaries (means):")
    for k in ["kl", "spearman", "pearson", "mse", "rmse", "mae", "r2", "ks", "wasserstein", "arfe", "mape", "argmax_err"]:
        if k in outputs.summary:
            print(f"  {k}: {outputs.summary[k]['mean']:.6f}")