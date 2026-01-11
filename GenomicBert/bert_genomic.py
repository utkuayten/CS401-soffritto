#!/usr/bin/env python3
# run_bert_genomic_save.py

import os
import json
import csv
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.data_loader import Dataset_Custom  # adjust if needed


class GenomicBERT(nn.Module):
    def __init__(
            self,
            enc_in: int = 9,
            c_out: int = 16,
            d_model: int = 256,
            n_heads: int = 4,
            num_layers: int = 4,
            d_ff: int = 512,
            dropout: float = 0.1,
            max_len: int = 512,
            rt2_idx: int = 8,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.rt2_idx = rt2_idx

        self.feat_proj = nn.Linear(enc_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        hidden_head = 128
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, hidden_head),
            nn.GELU(),
            nn.Linear(hidden_head, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if C != self.enc_in:
            raise ValueError(f"Expected enc_in={self.enc_in}, got {C}")

        rt2 = x[:, -1, self.rt2_idx:self.rt2_idx + 1]  # [B, 1]
        h = self.feat_proj(x)                          # [B, L, D]
        h = h + self.pos_embed[:, :L, :]               # [B, L, D]
        h_enc = self.encoder(h)                        # [B, L, D]
        h_last = h_enc[:, -1, :]                       # [B, D]
        h_cat = torch.cat([h_last, rt2], dim=-1)       # [B, D+1]
        logits = self.head(h_cat)                      # [B, 16]
        log_probs = F.log_softmax(logits, dim=-1)      # [B, 16]
        return log_probs.unsqueeze(1)                  # [B, 1, 16]


def build_loaders(
        root_path: str,
        data_path: str,
        train_chroms,
        val_chroms,
        seq_len=32,
        label_len=16,
        pred_len=1,
        batch_size=256,
        num_workers=4,
) -> Tuple[DataLoader, DataLoader]:
    size = [seq_len, label_len, pred_len]

    selected_cols = [
        "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
        "H3K4me3", "H3K9me3", "GC_content", "gene_density", "2-stage"
    ]

    train_ds = Dataset_Custom(
        root_path=root_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=val_chroms,
        flag="train",
        size=size,
        features="M",
        data_path=data_path,
        target="target_1",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="w",
        selected_cols=selected_cols,
    )

    val_ds = Dataset_Custom(
        root_path=root_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=val_chroms,
        flag="val",
        size=size,
        features="M",
        data_path=data_path,
        target="target_1",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="w",
        selected_cols=selected_cols,
    )

    # Pin memory helps GPU input pipeline
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    print("train len:", len(train_ds))
    print("val   len:", len(val_ds))
    return train_loader, val_loader


def ensure_prob_dist(target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Ensure target is a valid probability distribution along last dim.
    Use this ONLY if your labels are non-negative but not normalized.
    """
    target = torch.clamp(target, min=0.0)
    s = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return target / s


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        pred_len: int,
        normalize_target: bool,
        criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    n_batches = 0

    all_logp = []
    all_t = []

    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)

        target = batch_y[:, -pred_len:, :]  # [B, 1, 16]
        if normalize_target:
            target = ensure_prob_dist(target)

        log_probs = model(batch_x)          # [B, 1, 16]
        loss = criterion(log_probs, target)

        total += float(loss.item())
        n_batches += 1

        all_logp.append(log_probs.detach().cpu().numpy())
        all_t.append(target.detach().cpu().numpy())

    avg = total / max(1, n_batches)
    logp_np = np.concatenate(all_logp, axis=0)  # [N, 1, 16]
    t_np = np.concatenate(all_t, axis=0)        # [N, 1, 16]
    return avg, logp_np, t_np


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

class EarlyStopping:
    """
    Stop training when monitored metric (val_loss) does not improve.

    Args:
      patience: epochs to wait after last improvement
      min_delta: minimum change to qualify as improvement
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.num_bad = 0

    def step(self, current: float) -> bool:
        """
        Returns True if training should stop.
        """
        if current < (self.best - self.min_delta):
            self.best = current
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience

def main():
    # ======= CONFIG YOU NEED TO SET =======
    # Fix: do NOT put a leading "/" inside the relative path join
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Example: PROJECT_ROOT / "CS401-soffritto" / "GenomicBert" / "data"
    ROOT_PATH = str(PROJECT_ROOT / "CS401-soffritto" / "GenomicBert" / "data")
    DATA_PATH = "H1_genomic.csv"

    TRAIN_CHROMS = [1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
    VAL_CHROMS   = [6]

    seq_len   = 128
    label_len = 64
    pred_len  = 1
    enc_in    = 9
    c_out     = 16

    batch_size = 128
    epochs     = 15
    lr         = 0.0001711500076

    # Model hparams (yours)
    d_model = 512
    n_heads = 2
    num_layers = 6
    d_ff = 256
    dropout = 0.1812087669
    rt2_idx = 8

    # If your labels are NOT already probabilities, set True
    normalize_target = False

    # Early stopping
    early_stop_patience = 5      # change as you want
    early_stop_min_delta = 1e-4  # change as you want
    early_stopper = EarlyStopping(patience=early_stop_patience, min_delta=early_stop_min_delta)

    # Repro
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Output dir
    run_name = f"bert_genomic_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving to:", out_dir.resolve())

    # Save config
    config = dict(
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        train_chroms=TRAIN_CHROMS,
        val_chroms=VAL_CHROMS,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        enc_in=enc_in,
        c_out=c_out,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        rt2_idx=rt2_idx,
        normalize_target=normalize_target,
        seed=seed,
        device=str(device),
    )
    save_json(out_dir / "config.json", config)

    # Loaders
    train_loader, val_loader = build_loaders(
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        train_chroms=TRAIN_CHROMS,
        val_chroms=VAL_CHROMS,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        num_workers=4,
    )

    # Model
    model = GenomicBERT(
        enc_in=enc_in,
        c_out=c_out,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=seq_len,
        rt2_idx=rt2_idx,
    ).to(device)

    # Loss + opt
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # CSV logger
    csv_path = out_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"

    stopped_epoch = None

    # ====== TRAIN ======
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)

            target = batch_y[:, -pred_len:, :]  # [B, 1, 16]
            if normalize_target:
                target = ensure_prob_dist(target)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(batch_x)          # [B, 1, 16]
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_train = total_loss / max(1, len(train_loader))

        # ====== VALIDATION ======
        avg_val, _, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            pred_len=pred_len,
            normalize_target=normalize_target,
            criterion=criterion,
        )

        print(f"Epoch {epoch}: train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        # append to CSV
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train, avg_val])

        # save checkpoints
        torch.save(model.state_dict(), last_path)
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_path)

        # ---- EARLY STOPPING ----
        if early_stopper.step(avg_val):
            stopped_epoch = epoch
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best_val={early_stopper.best:.6f}, patience={early_stop_patience}, min_delta={early_stop_min_delta})."
            )
            break
    # ====== FINAL EVAL + SAVE ARRAYS ======
    # Load best model for exporting predictions
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_loss, logp_np, t_np = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        pred_len=pred_len,
        normalize_target=normalize_target,
        criterion=criterion,
    )

    # Convert to probabilities for "preds.npy"
    probs_np = np.exp(logp_np)  # since model outputs log_softmax

    # Argmax bins
    pred_bins = probs_np[:, 0, :].argmax(axis=-1)  # [N]
    true_bins = t_np[:, 0, :].argmax(axis=-1)      # [N]

    acc = float((pred_bins == true_bins).mean())

    np.save(out_dir / "preds.npy", probs_np)   # [N, 1, 16]
    np.save(out_dir / "trues.npy", t_np)       # [N, 1, 16]
    np.save(out_dir / "pred_bins.npy", pred_bins)
    np.save(out_dir / "true_bins.npy", true_bins)

    metrics = {
        "val_kl_div_loss": float(val_loss),
        "val_accuracy_argmax": acc,
        "n_val_samples": int(probs_np.shape[0]),
        "best_val_seen_during_training": float(best_val),
        "best_checkpoint": str(best_path.name),
    }
    save_json(out_dir / "metrics.json", metrics)

    print("Saved:")
    print(" -", best_path)
    print(" -", last_path)
    print(" -", out_dir / "preds.npy")
    print(" -", out_dir / "trues.npy")
    print(" -", out_dir / "metrics.json")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()