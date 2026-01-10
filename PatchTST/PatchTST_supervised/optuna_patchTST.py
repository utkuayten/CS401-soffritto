# optuna_patchtst.py
# Optuna hyperparameter tuning for PatchTST (genomic variant) using Exp_Main training pipeline.
#
# Saves:
#   optuna_results/<study_name>_partial.csv  (updated after each trial)
#   optuna_results/<study_name>.csv          (final)
#   optuna_results/<study_name>_best_params.json
#
# Usage example (single GPU):
#   python optuna_patchtst.py \
#       --root_path ./data \
#       --data_path H1_genomic.csv \
#       --cell H1 \
#       --train_chroms 1 2 3 4 5 7 8 10 11 12 13 14 15 16 17 18 19 20 21 22 \
#       --val_chroms 6 \
#       --test_chroms 9 \
#       --n_trials 30 \
#       --out_dir optuna_results

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, Optional, List

import optuna
import torch

# Import your existing training pipeline
from exp.exp_main import Exp_Main


def parse_args():
    p = argparse.ArgumentParser("Optuna tuning for PatchTST (genomic)")

    # Data / splits (match your run_longExp usage)
    p.add_argument("--root_path", type=str, default="./data")
    p.add_argument("--data_path", type=str, default="H1_genomic.csv")
    p.add_argument("--cell", type=str, default=None, help="If set, data_path becomes {cell}_genomic.csv under root_path")

    p.add_argument("--train_chroms", nargs="+", type=int, default=[1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22])
    p.add_argument("--val_chroms", nargs="+", type=int, default=[6])
    p.add_argument("--test_chroms", nargs="*", type=int, default=[9])

    # Core task sizes
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--label_len", type=int, default=16)
    p.add_argument("--pred_len", type=int, default=1)

    # Input/output dims (your genomic task)
    p.add_argument("--enc_in", type=int, default=9)
    p.add_argument("--c_out", type=int, default=16)

    # Training controls
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_amp", action="store_true", default=False)

    # Optuna controls
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_jobs", type=int, default=1, help="Keep 1 on a single GPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"])

    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///optuna_patchtst.db (resume support)")
    p.add_argument("--out_dir", type=str, default="optuna_results")

    # GPU
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use_gpu", type=int, default=1)

    return p.parse_args()


def ensure_list(x):
    if isinstance(x, int):
        return [x]
    return x


def build_base_args(args_cli) -> argparse.Namespace:
    """
    Create an argparse.Namespace compatible with Exp_Main / data_provider stack.
    Mirrors run_longExp.py defaults but only includes what PatchTST needs.
    """
    args = argparse.Namespace()

    # Repro / base
    args.random_seed = args_cli.seed
    args.is_training = 1
    args.model_id = "patchtst_optuna"
    args.model = "PatchTST"
    args.des = "optuna"

    # Data config
    args.data = "custom"
    args.root_path = args_cli.root_path
    args.data_path = args_cli.data_path
    if args_cli.cell:
        # If you want: data_path stored as full path OR filename; your pipeline expects root_path+data_path in provider.
        # Your run_longExp sets data_path to os.path.join(root_path, f"{cell}_genomic.csv") in one place.
        # Here we mimic: store full path if cell is provided.
        args.data_path = os.path.join(args.root_path, f"{args_cli.cell}_genomic.csv")

    args.features = "M"
    args.target = "target_1"
    args.freq = "w"
    args.embed = "timeF"
    args.checkpoints = "./checkpoints"

    # Splits (your genomic additions)
    args.cell = args_cli.cell
    args.train_chroms = ensure_list(args_cli.train_chroms)
    args.val_chroms = ensure_list(args_cli.val_chroms)
    args.test_chroms = ensure_list(args_cli.test_chroms)

    # Window sizes
    args.seq_len = int(args_cli.seq_len)
    args.label_len = int(args_cli.label_len)
    args.pred_len = int(args_cli.pred_len)

    # Dims
    args.enc_in = int(args_cli.enc_in)
    args.dec_in = 16
    args.c_out = int(args_cli.c_out)

    # PatchTST-specific defaults (will be overridden by Optuna)
    args.fc_dropout = 0.05
    args.head_dropout = 0.0
    args.patch_len = 2
    args.stride = 2
    args.padding_patch = "None"
    args.revin = 0
    args.affine = 0
    args.subtract_last = 1
    args.decomposition = 0

    args.kernel_size = 10
    args.individual = 0

    # Transformer params (shared)
    args.d_model = 128
    args.n_heads = 4
    args.e_layers = 3
    args.d_layers = 1
    args.d_ff = 512
    args.dropout = 0.1
    args.activation = "gelu"
    args.output_attention = False
    args.distil = False
    args.factor = 5
    args.attn = "prob"

    # Optimization
    args.num_workers = int(args_cli.num_workers)
    args.itr = 1
    args.train_epochs = int(args_cli.train_epochs)
    args.batch_size = 256
    args.patience = int(args_cli.patience)
    args.learning_rate = 1e-4
    args.lradj = "type1"
    args.pct_start = 0.3
    args.use_amp = bool(args_cli.use_amp)

    # GPU
    args.use_gpu = bool(args_cli.use_gpu) and torch.cuda.is_available()
    args.gpu = int(args_cli.gpu)
    args.use_multi_gpu = False
    args.devices = str(args.gpu)
    args.test_flop = False
    args.do_predict = False

    # Feature selection (keep consistent with your other code)
    args.selected_cols = [
        "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
        "H3K4me3", "H3K9me3", "GC_content", "gene_density", "2-stage"
    ]

    # Wavelet flags exist in your parser but can be no-ops here
    args.use_wavelet = False
    args.wavelet_name = "db4"
    args.wavelet_levels = 1
    args.keep_original = False
    args.wavelet_where = "dataset"

    # results path convenience
    args.results_path = "./results"

    return args


def tune_params(trial: optuna.Trial, args: argparse.Namespace) -> None:
    """
    Sample hyperparameters with Optuna and write into args (in-place).
    """

    # --- Architecture / PatchTST specifics ---
    args.patch_len = trial.suggest_categorical("patch_len", [2, 4, 8])
    args.stride = trial.suggest_categorical("stride", [1, 2, 4])

    # Constraint: patch_len <= seq_len and stride <= patch_len typically
    if args.patch_len > args.seq_len:
        raise optuna.TrialPruned(f"Invalid: patch_len({args.patch_len}) > seq_len({args.seq_len})")
    if args.stride <= 0 or args.stride > args.patch_len:
        raise optuna.TrialPruned(f"Invalid: stride({args.stride}) must be in [1..patch_len({args.patch_len})]")

    args.d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
    args.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])

    if args.d_model % args.n_heads != 0:
        raise optuna.TrialPruned(f"Invalid: d_model({args.d_model}) % n_heads({args.n_heads}) != 0")

    args.e_layers = trial.suggest_int("e_layers", 2, 6)
    args.d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024, 2048])

    args.dropout = trial.suggest_float("dropout", 0.05, 0.25)
    args.fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.25)
    args.head_dropout = trial.suggest_float("head_dropout", 0.0, 0.25)

    # RevIN / decomposition (optional toggles)
    args.revin = int(trial.suggest_categorical("revin", [0, 1]))
    args.subtract_last = int(trial.suggest_categorical("subtract_last", [0, 1]))
    args.decomposition = int(trial.suggest_categorical("decomposition", [0, 1]))
    if args.decomposition == 1:
        args.kernel_size = trial.suggest_categorical("kernel_size", [5, 9, 15, 25])

    # --- Training hyperparameters ---
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    # Note: Exp_Main uses Adam() without weight_decay. If you want wd, you must modify Exp_Main._select_optimizer.
    # We'll still record it for reproducibility, but it won't affect training unless you patch Exp_Main.
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


def compute_val_test_kl(exp: Exp_Main) -> Dict[str, float]:
    """
    Compute validation and test KL using exp.vali (same criterion as training).
    exp.train loads best checkpoint into exp.model before returning.
    """
    criterion = exp._select_criterion()

    val_data, val_loader = exp._get_data("val")
    test_data, test_loader = exp._get_data("test")

    val_kl = float(exp.vali(val_data, val_loader, criterion))
    test_kl = float(exp.vali(test_data, test_loader, criterion))
    return {"val_kl": val_kl, "test_kl": test_kl}


def save_study_csv(study: optuna.Study, out_csv: str) -> str:
    import pandas as pd
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete", "user_attrs"))
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    cli = parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    base_args = build_base_args(cli)

    if cli.study_name is None:
        val_str = "-".join(map(str, ensure_list(cli.val_chroms)))
        test_str = "-".join(map(str, ensure_list(cli.test_chroms))) if cli.test_chroms else "none"
        cli.study_name = f"optuna_patchtst_seq{cli.seq_len}_val{val_str}_test{test_str}"

    sampler = optuna.samplers.TPESampler(seed=cli.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    study = optuna.create_study(
        study_name=cli.study_name,
        direction=cli.direction,
        sampler=sampler,
        pruner=pruner,
        storage=cli.storage,
        load_if_exists=bool(cli.storage),
    )

    partial_csv = os.path.join(cli.out_dir, f"{cli.study_name}_partial.csv")
    final_csv = os.path.join(cli.out_dir, f"{cli.study_name}.csv")
    best_json = os.path.join(cli.out_dir, f"{cli.study_name}_best_params.json")

    def objective(trial: optuna.Trial) -> float:
        # Copy base args for this trial
        args = argparse.Namespace(**vars(base_args))

        # Apply trial params
        tune_params(trial, args)

        # Unique setting for checkpointing
        args.setting = f"{cli.study_name}_trial{trial.number}"
        args.checkpoints = os.path.join("./checkpoints", args.setting)

        # Train using your existing pipeline
        exp = Exp_Main(args)
        try:
            exp.train(args.setting)
            scores = compute_val_test_kl(exp)
            val_kl = scores["val_kl"]
            test_kl = scores["test_kl"]

            # Attach extras for CSV
            trial.set_user_attr("test_kl", test_kl)
            trial.set_user_attr("setting", args.setting)
            trial.set_user_attr("enc_in", args.enc_in)
            trial.set_user_attr("c_out", args.c_out)
            trial.set_user_attr("seq_len", args.seq_len)
            trial.set_user_attr("label_len", args.label_len)
            trial.set_user_attr("pred_len", args.pred_len)

            return val_kl

        except RuntimeError as e:
            msg = str(e).lower()
            # prune OOMs or CUDA issues
            if "out of memory" in msg or "cuda" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"Pruned due to runtime error: {e}")
            raise
        finally:
            # cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del exp

    def on_trial_complete(st: optuna.Study, tr: optuna.trial.FrozenTrial):
        # Write partial csv after each trial
        save_study_csv(st, partial_csv)

    study.optimize(
        objective,
        n_trials=cli.n_trials,
        n_jobs=cli.n_jobs,  # keep 1 on single GPU
        callbacks=[on_trial_complete],
        gc_after_trial=True,
        show_progress_bar=True,
    )

    save_study_csv(study, final_csv)

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("\n[✓] Optuna complete.")
    print("Best val KL:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved trials CSV:", final_csv)
    print("Saved best params:", best_json)


if __name__ == "__main__":
    main()