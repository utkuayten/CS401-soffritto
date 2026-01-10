# optuna_bert_genomic.py
#
# Optuna tuner for a BERT-like Transformer encoder on 9-channel genomic inputs
# predicting 16-fraction RT distributions with KLDiv loss.
#
# What it does:
# - Defines the GenomicBERT model (as in your run_bert_genomic.py)
# - Builds loaders using your existing Dataset_Custom
# - Runs Optuna hyperparameter tuning
# - Saves ALL trial results (params + objective + state + times) to CSV
# - Also saves best params to JSON
#
# Notes:
# - Keep n_jobs=1 if training on a single GPU.
# - This uses validation KL-div as the Optuna objective.
# - You can optionally evaluate on a held-out test chromosome and store test_KL as a user_attr.
#
# Usage example:
#   python optuna_bert_genomic.py --cell H1 --data_path H1_genomic.csv --root_path /path/to/data \
#       --train_chroms 1 2 3 ... --val_chroms 13 --test_chroms 9 --n_trials 30 --out_dir optuna_results
#
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PatchTST.PatchTST_supervised.data_provider.data_loader import PROJECT_ROOT
# Your dataset (as used by Informer code)
from data_loader.data_loader import Dataset_Custom


# -------------------------
# Model
# -------------------------
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
        # x: [B, L, enc_in]
        B, L, C = x.shape
        if C != self.enc_in:
            raise ValueError(f"Expected enc_in={self.enc_in}, got {C}")

        # scalar 2RT from last time step
        rt2 = x[:, -1, self.rt2_idx : self.rt2_idx + 1]  # [B, 1]

        h = self.feat_proj(x)  # [B, L, D]
        pos = self.pos_embed[:, :L, :]  # [1, L, D]
        h = h + pos

        h_enc = self.encoder(h)  # [B, L, D]
        h_last = h_enc[:, -1, :]  # [B, D]

        h_cat = torch.cat([h_last, rt2], dim=-1)  # [B, D+1]
        logits = self.head(h_cat)  # [B, 16]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, 16]
        return log_probs.unsqueeze(1)  # [B, 1, 16]


# -------------------------
# Data
# -------------------------
def build_loader(
        root_path: str,
        data_path: str,
        train_chroms,
        val_chroms,
        test_chroms,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        batch_size: int,
        num_workers: int,
) -> DataLoader:
    size = [seq_len, label_len, pred_len]
    ds = Dataset_Custom(
        root_path=root_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        flag=flag,
        size=size,
        features="M",
        data_path=data_path,
        target="target_1",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="w",
        selected_cols=[
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me3",
            "H3K9me3",
            "GC_content",
            "gene_density",
            "2-stage",
        ],
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(flag == "train"),
        num_workers=num_workers,
        drop_last=(flag == "train"),
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def build_loaders(
        root_path: str,
        data_path: str,
        train_chroms,
        val_chroms,
        test_chroms,
        seq_len: int,
        label_len: int,
        pred_len: int,
        batch_size: int,
        num_workers: int,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_loader = build_loader(
        root_path, data_path, train_chroms, val_chroms, test_chroms,
        flag="train", seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers
    )
    val_loader = build_loader(
        root_path, data_path, train_chroms, val_chroms, test_chroms,
        flag="val", seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers
    )

    # Optional test loader (if you want a real held-out chromosome evaluation)
    test_loader = None
    if test_chroms and len(test_chroms) > 0:
        test_loader = build_loader(
            root_path, data_path, train_chroms, val_chroms, test_chroms,
            flag="test", seq_len=seq_len, label_len=label_len, pred_len=pred_len,
            batch_size=batch_size, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def eval_kl(model: nn.Module, loader: DataLoader, device: torch.device, pred_len: int) -> float:
    model.eval()
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)

    total = 0.0
    n_batches = 0
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        target = batch_y[:, -pred_len:, :]  # [B, 1, 16]

        log_probs = model(batch_x)  # [B, 1, 16]
        loss = criterion(log_probs, target)
        total += float(loss.item())
        n_batches += 1

    return total / max(1, n_batches)


def train_one_trial(
        trial: optuna.Trial,
        root_path: str,
        data_path: str,
        train_chroms,
        val_chroms,
        test_chroms,
        enc_in: int,
        c_out: int,
        rt2_idx: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        device: torch.device,
        max_epochs: int,
        patience: int,
        num_workers: int,
) -> Tuple[float, Dict[str, Any]]:
    # --- hyperparameters to tune ---
    seq_len = trial.suggest_categorical("seq_len", [32, 64, 128])
    label_len = seq_len // 2
    d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # constraint: d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        raise optuna.TrialPruned(f"Invalid: d_model({d_model}) % n_heads({n_heads}) != 0")

    train_loader, val_loader, test_loader = build_loaders(
        root_path=root_path,
        data_path=data_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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

    criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad = 0

    # training loop with early stopping
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device, non_blocking=True)
            batch_y = batch_y.float().to(device, non_blocking=True)
            target = batch_y[:, -pred_len:, :]  # [B, 1, 16]

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(batch_x)
            loss = criterion(log_probs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        # validation
        val_kl = eval_kl(model, val_loader, device, pred_len=pred_len)

        # report intermediate value to Optuna (enables pruning)
        trial.report(val_kl, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with val_kl={val_kl:.6f}")

        if val_kl < best_val - 1e-6:
            best_val = val_kl
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    # optional test evaluation
    test_kl = None
    if test_loader is not None:
        test_kl = eval_kl(model, test_loader, device, pred_len=pred_len)

    extras = {
        "best_epoch": best_epoch,
        "test_kl": float(test_kl) if test_kl is not None else None,
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
    }

    # aggressive cleanup (important for Optuna loops)
    del model
    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return best_val, extras


# -------------------------
# Optuna runner
# -------------------------
def parse_args():
    p = argparse.ArgumentParser("Optuna tuning for GenomicBERT")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    p.add_argument("--root_path", type=str, default=str(PROJECT_ROOT / "GenomicBert/data/"), help="Directory containing the genomic CSV.")
    p.add_argument("--data_path", type=str, default="H1_genomic.csv", help="CSV filename, e.g., H1_genomic.csv")

    p.add_argument("--train_chroms", nargs="+", type=int, default=[1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22])
    p.add_argument("--val_chroms", nargs="+", type=int, default=[6])
    p.add_argument("--test_chroms", nargs="*", type=int, default=[9])

    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--label_len", type=int, default=16)
    p.add_argument("--pred_len", type=int, default=1)

    p.add_argument("--enc_in", type=int, default=9)
    p.add_argument("--c_out", type=int, default=16)
    p.add_argument("--rt2_idx", type=int, default=8)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--n_jobs", type=int, default=1, help="Keep 1 on a single GPU.")

    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///optuna_bert.db (resume support)")
    p.add_argument("--out_dir", type=str, default="optuna_results")

    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def save_study_csv(study: optuna.Study, csv_path: str) -> str:
    import pandas as pd
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete", "user_attrs"))
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.study_name is None:
        args.study_name = f"optuna_genomicbert_seq{args.seq_len}_val{'-'.join(map(str,args.val_chroms))}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: optuna.Trial) -> float:
        best_val_kl, extras = train_one_trial(
            trial=trial,
            root_path=args.root_path,
            data_path=args.data_path,
            train_chroms=args.train_chroms,
            val_chroms=args.val_chroms,
            test_chroms=args.test_chroms,
            enc_in=args.enc_in,
            c_out=args.c_out,
            rt2_idx=args.rt2_idx,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
        )
        # store extras to CSV via user_attrs
        for k, v in extras.items():
            trial.set_user_attr(k, v)
        return best_val_kl

    # optional: crash-safe periodic CSV
    partial_csv = os.path.join(args.out_dir, f"{args.study_name}_partial.csv")

    def on_trial_complete(st: optuna.Study, tr: optuna.trial.FrozenTrial):
        save_study_csv(st, partial_csv)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=[on_trial_complete],
        gc_after_trial=True,
        show_progress_bar=True,
    )

    final_csv = os.path.join(args.out_dir, f"{args.study_name}.csv")
    save_study_csv(study, final_csv)

    best_json = os.path.join(args.out_dir, f"{args.study_name}_best_params.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("\n[✓] Done.")
    print("Best value (val KL):", study.best_value)
    print("Best params:", study.best_params)
    print("Saved trials CSV:", final_csv)
    print("Saved best params:", best_json)


if __name__ == "__main__":
    main()