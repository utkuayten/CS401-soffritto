# optuna_intra_cell.py
# Optuna hyperparameter tuning wrapper for train_intra_cell.py (fixed val_chrom, fixed test_chrom)
#
# Usage example:
#   python optuna_intra_cell.py --cell mESC --test_chrom 1 --val_chrom 6 --n_trials 30 --study_name mESC_test1_val6
#
# Notes:
# - Do NOT run GPU trials in parallel on a single GPU. Default n_jobs=1.
# - This script saves trials to CSV (partial + final).
# - Objective is minimized by default (e.g., KL divergence). You can switch to maximize if needed.

import os

# Must be set BEFORE torch is imported anywhere (train code will import torch internally).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

import argparse
import gc
from argparse import Namespace
from typing import Any, Dict, Optional

import optuna

from train_intra_cell import main as train_intra_cell_main


def parse_args():
    p = argparse.ArgumentParser("Optuna tuning for intra-cell Informer training (fixed val_chrom).")

    # Data split controls
    p.add_argument("--cell", type=str, required=True, help="Cell name (e.g., mESC or H1)")
    p.add_argument("--test_chrom", type=int, required=True, help="Chromosome held out for testing")
    p.add_argument("--val_chrom", type=int, default=6, help="Single validation chromosome (fixed)")

    # Optuna controls
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--n_jobs", type=int, default=1, help="Parallel trials. Keep 1 on single GPU.")
    p.add_argument("--seed", type=int, default=42)

    # Persist/Resume (recommended)
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g., sqlite:///optuna_intra_cell.db (enables resume).",
    )

    # CSV output
    p.add_argument("--out_dir", type=str, default="optuna_results")
    p.add_argument("--save_every_trial", action="store_true", help="Write partial CSV after each trial.")

    # Training resource knobs (hard caps to prevent OOM)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_epochs", type=int, default=10, help="Upper bound; actual value will be tuned <= this.")
    p.add_argument("--patience", type=int, default=3)

    # Metric selection from train_intra_cell_main return dict
    p.add_argument(
        "--objective_key",
        type=str,
        default=None,
        help=(
            "Metric key to optimize from returned metrics dict (e.g., 'val_score', 'kl', 'loss'). "
            "If not set, script will try common keys."
        ),
    )

    # Optional feature config passthrough
    p.add_argument(
        "--selected_cols",
        nargs="+",
        type=str,
        default=["H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3", "GC_content", "gene_density", "2-stage"],
    )

    # Wavelet passthrough (also tunable via Optuna)
    p.add_argument("--allow_wavelet", action="store_true", help="Allow Optuna to toggle wavelet features.")

    return p.parse_args()


def get_all_chroms(cell: str):
    # mESC has chr1-19, human-like has chr1-22 (as in your earlier script)
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 23))


def pick_objective_value(metrics: Any, objective_key: Optional[str] = None) -> float:
    """
    Extract a numeric objective from train_intra_cell_main's return value.
    - If objective_key is provided, use it.
    - Otherwise try common keys.
    """
    if metrics is None:
        return float("inf")

    if isinstance(metrics, (int, float)):
        return float(metrics)

    if not isinstance(metrics, dict):
        return float("inf")

    if objective_key:
        v = metrics.get(objective_key, None)
        return float(v) if v is not None else float("inf")

    # Common fallbacks (edit to match your run_model_main outputs)
    for k in ["val_score", "val_loss", "kl", "KL", "loss", "test_loss", "metric", "score"]:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])

    return float("inf")


def cleanup_cuda():
    # Make cleanup robust even if torch isn't available in environment.
    try:
        import torch  # noqa: F401
        import torch.cuda

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        gc.collect()


def build_trial_args(
        trial: optuna.Trial,
        base: argparse.Namespace,
        train_chroms,
        val_chrom,
        test_chrom,
) -> Namespace:
    # Core sequence hyperparameters
    seq_len = trial.suggest_categorical("seq_len", [32, 64, 96, 128])
    label_len = seq_len // 2

    # Architecture hyperparameters
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])

    # Ensure divisibility: d_model must be divisible by n_heads
    if d_model % n_heads != 0:
        # Penalize invalid configs
        raise optuna.TrialPruned(f"Invalid config: d_model({d_model}) % n_heads({n_heads}) != 0")

    args = Namespace(
        # Required split args for train_intra_cell.py
        cell=base.cell,
        train_chroms=train_chroms,
        val_chroms=[val_chrom],
        test_chroms=[test_chrom],  # IMPORTANT: test_chrom, not val_chrom
        setting=None,
        # Sequence
        seq_len=seq_len,
        label_len=label_len,
        pred_len=1,
        # Architecture
        enc_in=trial.suggest_categorical("enc_in", [9]),     # keep fixed unless you truly vary features
        dec_in=trial.suggest_categorical("dec_in", [16]),
        c_out=trial.suggest_categorical("c_out", [16]),
        e_layers=trial.suggest_int("e_layers", 1, 4),
        d_layers=trial.suggest_int("d_layers", 1, 4),
        d_model=d_model,
        n_heads=n_heads,
        d_ff=trial.suggest_categorical("d_ff", [256, 512, 1024, 2048]),
        dropout=trial.suggest_float("dropout", 0.01, 0.2),
        attn=trial.suggest_categorical("attn", ["prob", "full"]),
        factor=trial.suggest_categorical("factor", [3, 5, 7]),
        activation=trial.suggest_categorical("activation", ["gelu", "relu"]),
        # Training
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        train_epochs=trial.suggest_int("train_epochs", 3, base.max_train_epochs),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        patience=base.patience,
        lradj=trial.suggest_categorical("lradj", ["type1", "type2", "type3"]),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        num_workers=base.num_workers,
        # GPU
        use_multi_gpu=False,
        gpu=base.gpu,
        devices=str(base.gpu),
        # Feature selection
        selected_cols=base.selected_cols,
        # Wavelet (optionally tunable)
        use_wavelet=False,
        wavelet_name="db4",
        wavelet_levels=1,
        keep_original=False,
        wavelet_where="dataset",
    )

    if base.allow_wavelet:
        args.use_wavelet = trial.suggest_categorical("use_wavelet", [False, True])
        if args.use_wavelet:
            args.wavelet_name = trial.suggest_categorical("wavelet_name", ["db4", "coif1", "sym4"])
            args.wavelet_levels = trial.suggest_int("wavelet_levels", 1, 2)
            args.keep_original = trial.suggest_categorical("keep_original", [False, True])
            args.wavelet_where = trial.suggest_categorical("wavelet_where", ["dataset", "model"])

    # Create a unique setting name per trial
    args.setting = f"{base.cell}_val{val_chrom}_test{test_chrom}_trial{trial.number}"

    # Keep checkpoints separated per trial (train_intra_cell.py will also derive paths)
    args.checkpoints = os.path.join("checkpoints", args.setting)

    return args


def objective_factory(base: argparse.Namespace):
    chroms_all = get_all_chroms(base.cell)

    if base.val_chrom == base.test_chrom:
        raise ValueError("val_chrom and test_chrom must be different.")

    if base.val_chrom not in chroms_all:
        raise ValueError(f"val_chrom={base.val_chrom} is not valid for cell={base.cell}")

    if base.test_chrom not in chroms_all:
        raise ValueError(f"test_chrom={base.test_chrom} is not valid for cell={base.cell}")

    train_chroms = [c for c in chroms_all if c not in (base.val_chrom, base.test_chrom)]

    def _objective(trial: optuna.Trial) -> float:
        args = build_trial_args(trial, base, train_chroms, base.val_chrom, base.test_chrom)

        try:
            metrics = train_intra_cell_main(args)
        except RuntimeError as e:
            # Catch CUDA OOM and prune the trial instead of killing the whole study
            msg = str(e).lower()
            if "cuda out of memory" in msg or "cublas" in msg:
                cleanup_cuda()
                raise optuna.TrialPruned(f"Pruned due to runtime error: {e}")
            cleanup_cuda()
            raise

        # Store extra info in Optuna
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                # Only store JSON-serializable simple types
                if isinstance(v, (int, float, str, bool)) or v is None:
                    trial.set_user_attr(k, v)

        value = pick_objective_value(metrics, base.objective_key)

        cleanup_cuda()
        return value

    return _objective


def save_study_csv(study: optuna.Study, csv_path: str):
    import pandas as pd

    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete", "user_attrs"))
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.study_name is None:
        args.study_name = f"optuna_{args.cell}_val{args.val_chrom}_test{args.test_chrom}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    csv_partial = os.path.join(args.out_dir, f"{args.study_name}_partial.csv")
    csv_final = os.path.join(args.out_dir, f"{args.study_name}.csv")

    callbacks = []
    if args.save_every_trial:

        def _cb(st: optuna.Study, tr: optuna.trial.FrozenTrial):
            save_study_csv(st, csv_partial)

        callbacks.append(_cb)

    objective = objective_factory(args)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,  # Keep 1 on single GPU
        callbacks=callbacks if callbacks else None,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # Save final CSV
    save_study_csv(study, csv_final)

    print("\n[✓] Best trial:")
    print(f"  Value:  {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    print(f"\n[✓] Saved Optuna trials to: {csv_final}")
    if args.save_every_trial:
        print(f"[✓] Partial CSV path: {csv_partial}")


if __name__ == "__main__":
    main()