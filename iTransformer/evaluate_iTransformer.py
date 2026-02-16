# evaluator.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from sklearn.metrics import r2_score, confusion_matrix

import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray]


def _safe_normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if np.isnan(p).any():
        raise ValueError("Input contains NaNs.")

    row_sum = p.sum(axis=-1, keepdims=True)

    # Case A: already probabilities
    if (p >= -1e-12).all() and np.allclose(row_sum, 1.0, atol=1e-3):
        p = np.clip(p, eps, 1.0)
        p = p / p.sum(axis=-1, keepdims=True)
        return p

    # Case B: logits or log-probs -> softmax
    x = p - p.max(axis=-1, keepdims=True)
    probs = np.exp(np.clip(x, -80, 80))
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


def _kl_divergence_batch(true_p: np.ndarray, pred_p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Q = np.clip(true_p, eps, 1.0)
    P = np.clip(pred_p, eps, 1.0)
    return np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)


def _cdf_ks_discrete(true_p: np.ndarray, pred_p: np.ndarray) -> np.ndarray:
    cdf_t = np.cumsum(true_p, axis=-1)
    cdf_p = np.cumsum(pred_p, axis=-1)
    return np.max(np.abs(cdf_t - cdf_p), axis=-1)


def _wasserstein_1d_discrete(true_p: np.ndarray, pred_p: np.ndarray) -> np.ndarray:
    n, c = true_p.shape
    support = np.arange(c, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = wasserstein_distance(support, support, u_weights=true_p[i], v_weights=pred_p[i])
    return out


def _vector_metrics(true_p: np.ndarray, pred_p: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    diff = pred_p - true_p
    abs_diff = np.abs(diff)

    mse = np.mean(diff ** 2, axis=-1)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff, axis=-1)

    mape = np.mean(abs_diff / np.clip(np.abs(true_p), eps, None), axis=-1) * 100.0
    arfe = np.mean(abs_diff / np.clip(true_p, eps, None), axis=-1)

    spearman = np.empty(true_p.shape[0], dtype=np.float64)
    pearson = np.empty(true_p.shape[0], dtype=np.float64)
    for i in range(true_p.shape[0]):
        rho, _ = spearmanr(true_p[i], pred_p[i])
        spearman[i] = rho
        r, _ = pearsonr(true_p[i], pred_p[i])
        pearson[i] = r

    r2 = np.empty(true_p.shape[0], dtype=np.float64)
    for i in range(true_p.shape[0]):
        r2[i] = r2_score(true_p[i], pred_p[i])

    ks = _cdf_ks_discrete(true_p, pred_p)
    wdist = _wasserstein_1d_discrete(true_p, pred_p)

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
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.nanmean(x)),
        "median": float(np.nanmedian(x)),
        "std": float(np.nanstd(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }


def _to_2d(a: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (N,C)
      - (N,1,C)
      - (N,T,C)
      - (N,1,1,C) or any (....,C)
      - (N,) integer class labels (will be handled elsewhere)
    Returns:
      - (N,C) or (N*T,C) by flattening all dims except the last.
    """
    a = np.asarray(a)

    # If it's a vector of class indices, keep it 1D (handled in _coerce_labels below)
    if a.ndim == 1:
        return a

    if a.ndim < 2:
        raise ValueError(f"Expected at least 2D array with class dimension last, got shape={a.shape}")

    # Flatten all leading dims into one, keep last dim as C
    c = a.shape[-1]
    a = a.reshape(-1, c)

    if a.ndim != 2:
        raise ValueError(f"Expected 2D after reshape, got {a.shape}")
    return a

def _labels_to_onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(np.int64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D labels, got {y.shape}")
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError(f"Label values out of range [0, {num_classes-1}]")
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out
@dataclass
class EvaluationOutputs:
    summary: Dict[str, Dict[str, float]]
    per_bin: Dict[str, np.ndarray]
    confusion: np.ndarray
    paths: Dict[str, str]


class ReplicationTimingEvaluator:
    def __init__(self, out_dir: str = "eval_outputs", class_names: Optional[list] = None, eps: float = 1e-12):
        self.out_dir = out_dir
        self.eps = eps
        self.class_names = class_names or [str(i) for i in range(16)]
        os.makedirs(self.out_dir, exist_ok=True)

    def load_from_npy(self, pred_path: str, true_path: str) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.load(pred_path)
        true = np.load(true_path)
        return pred, true

    def load_other_predictions(self, path: str, npz_key: Optional[str] = None) -> np.ndarray:
        """
        Loads other model predictions from:
          - .npy (np.load)
          - .npz (np.load, then pick npz_key if given else first array)
        """
        path = str(path)
        if path.endswith(".npy"):
            return np.load(path)

        if path.endswith(".npz"):
            z = np.load(path)
            if npz_key is not None:
                if npz_key not in z.files:
                    raise KeyError(f"npz_key='{npz_key}' not found. Available keys: {z.files}")
                return z[npz_key]
            if len(z.files) == 0:
                raise ValueError("NPZ contains no arrays.")
            return z[z.files[0]]

        raise ValueError(f"Unsupported file type for other predictions: {path} (use .npy or .npz)")

    def evaluate(self, pred: ArrayLike, true: ArrayLike, prefix: str = "test", save_per_bin_csv: bool = True) -> EvaluationOutputs:
        pred = _to_2d(np.asarray(pred))
        true = _to_2d(np.asarray(true))

        # If true is labels (N,), convert to one-hot using number of classes inferred from pred
        if true.ndim == 1:
            if pred.ndim != 2:
                raise ValueError(f"Cannot infer num_classes from pred shape={pred.shape}")
            true = _labels_to_onehot(true, num_classes=pred.shape[1])

        # If pred is labels (rare, but possible), convert to one-hot too
        if pred.ndim == 1:
            pred = _labels_to_onehot(pred, num_classes=len(self.class_names))

        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred{pred.shape}, true{true.shape}")

        true_p = _safe_normalize_probs(true, eps=self.eps)
        pred_p = _safe_normalize_probs(pred, eps=self.eps)

        kl = _kl_divergence_batch(true_p, pred_p, eps=self.eps)

        per_bin = _vector_metrics(true_p, pred_p, eps=self.eps)
        per_bin["kl"] = kl

        y_true = per_bin["true_argmax"]
        y_pred = per_bin["pred_argmax"]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))

        summary: Dict[str, Dict[str, float]] = {}
        for k, v in per_bin.items():
            if k in ("true_argmax", "pred_argmax"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 1:
                summary[k] = _nan_safe_agg(v)

        paths: Dict[str, str] = {}

        paths["summary_json"] = os.path.join(self.out_dir, f"{prefix}_summary_metrics.json")
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        paths["confusion_csv"] = os.path.join(self.out_dir, f"{prefix}_confusion_matrix.csv")
        np.savetxt(paths["confusion_csv"], cm, delimiter=",", fmt="%d")

        paths["confusion_png"] = os.path.join(self.out_dir, f"{prefix}_confusion_matrix.png")
        self._save_confusion_plot(cm, paths["confusion_png"], title=f"Confusion Matrix ({prefix})")

        if save_per_bin_csv:
            paths["per_bin_csv"] = os.path.join(self.out_dir, f"{prefix}_per_bin_metrics.csv")
            self._save_per_bin_csv(per_bin, paths["per_bin_csv"])

        return EvaluationOutputs(summary=summary, per_bin=per_bin, confusion=cm, paths=paths)

    def _save_per_bin_csv(self, per_bin: Dict[str, np.ndarray], out_path: str) -> None:
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

        im = ax.imshow(cm, aspect="auto", cmap="gray_r")

        ax.set_title(title)
        ax.set_xlabel("Predicted argmax fraction")
        ax.set_ylabel("True argmax fraction")

        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        vmax = cm.max() if cm.size else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                text_color = "black" if val > (0.5 * vmax) else "white"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=text_color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def save_three_heatmaps(
            self,
            true: np.ndarray,
            pred: np.ndarray,
            other_pred: np.ndarray,
            out_path: str,
            title: str = "Heatmaps: True vs Our Pred vs Other Pred",
            other_label: str = "Other model",
    ) -> str:
        """
        Creates ONE figure with THREE panels:
          - True
          - Our predictions
          - Other model predictions
        Includes the intensity colorbar on the figure.
        No sorting, no row limits.
        """
        true2 = _to_2d(true)
        pred2 = _to_2d(pred)
        other2 = _to_2d(other_pred)

        if true2.shape != pred2.shape or true2.shape != other2.shape:
            raise ValueError(
                f"Shape mismatch:\n  true{true2.shape}\n  pred{pred2.shape}\n  other{other2.shape}"
            )

        true_p = _safe_normalize_probs(true2, eps=self.eps)
        pred_p = _safe_normalize_probs(pred2, eps=self.eps)
        other_p = _safe_normalize_probs(other2, eps=self.eps)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

        im1 = ax1.imshow(true_p, aspect="auto", cmap="gray_r", interpolation="nearest")
        ax1.set_title("True")
        ax1.set_xlabel("S-phase fraction (1..16)")
        ax1.set_ylabel("Bin index")

        im2 = ax2.imshow(pred_p, aspect="auto", cmap="gray_r", interpolation="nearest")
        ax2.set_title("Predicted (ours)")
        ax2.set_xlabel("S-phase fraction (1..16)")
        ax2.set_ylabel("Bin index")

        im3 = ax3.imshow(other_p, aspect="auto", cmap="gray_r", interpolation="nearest")
        ax3.set_title(f"Predicted ({other_label})")
        ax3.set_xlabel("S-phase fraction (1..16)")
        ax3.set_ylabel("Bin index")

        c = true_p.shape[1]
        ticks = np.arange(c)
        labels = [str(i + 1) for i in range(c)]
        for ax in (ax1, ax2, ax3):
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

        fig.suptitle(title)

        # One shared intensity stick on the figure
        cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], location="right", shrink=0.95, pad=0.02)
        cbar.set_label("Probability")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

def align_to_true(true2: np.ndarray, pred2: np.ndarray, other2: np.ndarray):
    n = true2.shape[0]
    pred2 = pred2[:n]
    other2 = other2[:n]
    return true2, pred2, other2

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Our model outputs
    pred_path = str(PROJECT_ROOT / "transoffritto/iTransformer/results/H1_val6_test9/pred.npy")
    true_path = str(PROJECT_ROOT / "transoffritto/iTransformer/results/H1_val6_test9/true.npy")

    # Other model predictions (your attached file is a .npy)
    #other_pred_path = str(PROJECT_ROOT / "transoffritto/iTransformer/results/H1_val6_test9/H1_chr9_pred_intra_cell_line.npy")
    # If it's elsewhere, set an absolute path instead.

    evaluator = ReplicationTimingEvaluator(out_dir="eval_outputs", class_names=[str(i) for i in range(16)])

    pred, true = evaluator.load_from_npy(pred_path, true_path)
    #other_pred = evaluator.load_other_predictions(other_pred_path)

    true, pred, other_pred = align_to_true(true, pred, pred)

    # Metrics for our model
    outputs_ours = evaluator.evaluate(pred, true, prefix="chrom_test_ours", save_per_bin_csv=True)

    # Metrics for other model (optional but typically useful)
    outputs_other = evaluator.evaluate(other_pred, true, prefix="chrom_test_other", save_per_bin_csv=True)

    # 3-panel heatmap figure: True vs Our Pred vs Other Pred
    heatmap_path = os.path.join(evaluator.out_dir, "chrom_test_true_vs_ours_vs_other_heatmap.png")
    evaluator.save_three_heatmaps(
        true=true,
        pred=pred,
        other_pred=other_pred,
        out_path=heatmap_path,
        title="Chromosome 9 (H1): True vs Pred (ours) vs Pred (other)",
        other_label="Soffritto",
    )
    print("Saved 3-panel heatmap:", heatmap_path)

    print("\nSaved (ours):")
    for k, v in outputs_ours.paths.items():
        print(f"  {k}: {v}")

    print("\nSaved (other):")
    for k, v in outputs_other.paths.items():
        print(f"  {k}: {v}")