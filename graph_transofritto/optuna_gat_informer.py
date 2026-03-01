# optuna_tune_intra_cell.py
"""
Optuna tuner for GAT+Informer architecture hyperparameters.

How it works
- Builds an argparse-style Namespace (same fields your train_intra_cell.py creates)
- Calls run_model_main(args) for each trial
- Minimizes validation metric returned by run_model_main (you can adapt key selection below)

Usage
  python3 optuna_tune_intra_cell.py --cell H1 --n_trials 50

Notes
- Ensures d_model % n_heads == 0
- Tunes ONLY model/architecture by default; you can also tune lr/weight_decay if you want.
"""

import os
import json
import math
import argparse
from types import SimpleNamespace

import optuna

from run_model import run_model_main


class OptunaGATInformerTuner:
    """
    Full tuner "class" that you can run as a script or import.
    """

    def __init__(
            self,
            cell: str = "H1",
            train_chroms=None,
            val_chroms=None,
            test_chroms=None,
            seq_len: int = 32,
            label_len: int = 16,
            pred_len: int = 1,
            selected_cols=None,
            decoding_mode: str = "teacher-forced",
            rt2_col: str = "2-stage",
            # infra paths (relative to this file by default)
            base_dir: str = None,
            # optuna
            study_name: str = "gat_informer_arch_tuning",
            storage: str = None,  # e.g. "sqlite:///optuna_gatinformer.db"
            direction: str = "minimize",
            sampler_seed: int = 42,
    ):
        self.cell = cell
        self.train_chroms = train_chroms or [1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
        self.val_chroms = val_chroms or [6]
        self.test_chroms = test_chroms or [9]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.selected_cols = selected_cols or [
            'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
            'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'
        ]

        # IMPORTANT: use the exact strings your pipeline expects
        # Your argparse "choices" are ["teacher-forced", "cost-aware-1", "cost-aware-2"]
        if decoding_mode not in ["teacher-forced", "cost-aware-1", "cost-aware-2"]:
            raise ValueError(f"decoding_mode must be one of teacher-forced/cost-aware-1/cost-aware-2, got: {decoding_mode}")
        self.decoding_mode = decoding_mode
        self.rt2_col = rt2_col

        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.sampler_seed = sampler_seed

        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        self.base_dir = base_dir

        self.root_path = os.path.join(self.base_dir, "data")
        self.data_path = os.path.join(self.root_path, f"{self.cell}_genomic.csv")
        self.checkpoints = os.path.join(self.base_dir, "checkpoints")
        self.results_path = os.path.join(self.base_dir, "results")

        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, n_trials: int = 50, timeout: int = None, n_jobs: int = 1):
        sampler = optuna.samplers.TPESampler(seed=self.sampler_seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        best = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "study_name": self.study_name,
            "direction": self.direction,
            "n_trials": len(study.trials),
        }
        out_path = os.path.join(self.results_path, f"{self.study_name}_best.json")
        with open(out_path, "w") as f:
            json.dump(best, f, indent=2)

        print("\n=== Optuna Finished ===")
        print(json.dumps(best, indent=2))
        print(f"[Saved] {out_path}")
        return study

    # ----------------------------
    # Core logic
    # ----------------------------

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Returns the validation score to MINIMIZE.
        Adapt `score = ...` selection to match your run_model_main output format.
        """

        args = self._build_args_for_trial(trial)

        # Run training/eval for this trial
        metrics = run_model_main(args)

        # ---- Decide what to optimize ----
        # Your run_model_main seems to return "metrics" (maybe dict or tuple).
        # Common patterns handled below:
        #  1) dict with 'val'/'valid'/'val_KL'/'best' keys
        #  2) single float
        #  3) tuple/list where first element is val score
        score = self._extract_score(metrics)

        # report to optuna + allow pruning
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def _build_args_for_trial(self, trial: optuna.trial.Trial):
        """
        Creates a SimpleNamespace with exactly the fields your pipeline expects.
        Fixes Optuna dynamic categorical-space issue by using static choices and pruning invalid combos.
        """
        from types import SimpleNamespace
        args = SimpleNamespace()

        # --------- Fixed / dataset config ----------
        args.setting = None
        args.cell = self.cell
        args.train_chroms = list(self.train_chroms)
        args.val_chroms = list(self.val_chroms)
        args.test_chroms = list(self.test_chroms)

        args.seq_len = self.seq_len
        args.label_len = self.label_len
        args.pred_len = self.pred_len

        args.enc_in = len(self.selected_cols)  # typically 9
        args.dec_in = 16
        args.c_out = 16

        # --------- Tuned: Informer architecture (STATIC SPACES) ----------
        # d_model from a static list
        d_model = trial.suggest_categorical("d_model", [64, 96, 128, 160, 192, 256, 384, 512])

        # n_heads from a static list, then PRUNE if invalid
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16])
        if d_model % n_heads != 0:
            raise optuna.TrialPruned(f"Invalid combo: d_model={d_model} not divisible by n_heads={n_heads}")

        args.d_model = d_model
        args.n_heads = n_heads

        args.e_layers = trial.suggest_int("e_layers", 2, 6)
        args.d_layers = trial.suggest_int("d_layers", 2, 6)

        # ---- Option A (recommended): tune FF multiplier (STATIC), set d_ff = mult * d_model ----
        ff_mult = trial.suggest_categorical("ff_mult", [2, 4, 6, 8])
        d_ff = ff_mult * d_model
        # clip to keep it sane
        d_ff = int(max(256, min(d_ff, 4096)))
        args.d_ff = d_ff

        args.dropout = trial.suggest_float("dropout", 0.0, 0.2)
        args.attn = trial.suggest_categorical("attn", ["full", "prob"])
        args.factor = trial.suggest_int("factor", 3, 10)
        args.activation = trial.suggest_categorical("activation", ["relu", "gelu", "elu"])

        # --------- Tuned: GAT front-end ----------
        args.model = "gatinformer"
        args.gat_layers = trial.suggest_int("gat_layers", 1, 4)
        args.gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 3, 4, 6, 8])
        args.gat_k = trial.suggest_int("gat_k", 1, 8)
        args.gat_hidden = trial.suggest_categorical("gat_hidden", [None, args.enc_in, 16, 32, 64])
        args.gat_dropout = trial.suggest_float("gat_dropout", 0.0, 0.2)
        args.apply_gat_to_dec = trial.suggest_categorical("apply_gat_to_dec", [False, True])

        # --------- (Optional) optimizer knobs ----------
        args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)

        # --------- Training loop controls ----------
        args.train_epochs = 10
        args.batch_size = 512
        args.patience = 3
        args.lradj = "type3"
        args.num_workers = 1

        # GPU/MPS
        args.use_multi_gpu = False
        args.gpu = 0
        args.devices = "0"

        # Wavelet off for tuning (unless you want it)
        args.use_wavelet = False
        args.wavelet_name = "db4"
        args.wavelet_levels = 1
        args.keep_original = False
        args.wavelet_where = "dataset"

        # Feature selection + decoding mode
        args.selected_cols = list(self.selected_cols)
        args.decoding_mode = self.decoding_mode
        args.rt2_col = self.rt2_col

        # Constant config expected by run_model_main / Exp_Informer
        args.root_path = self.root_path
        args.data_path = self.data_path
        args.checkpoints = self.checkpoints
        args.results_path = self.results_path

        args.target = "target_1"
        args.freq = "w"
        args.embed = "timeF"
        args.output_attention = False
        args.distil = False
        args.mix = False
        args.data = "custom"
        args.features = "M"
        args.inverse = False
        args.padding = 0

        # Setting string
        if not args.setting:
            val_str = "-".join(str(c) for c in args.val_chroms)
            args.setting = f"{args.cell}_val_{val_str}_trial_{trial.number}"

        return args

    def _extract_score(self, metrics):
        """
        Robustly extract a scalar score from run_model_main output.

        Adjust this function if your run_model_main returns a different structure.
        """
        # dict case
        if isinstance(metrics, dict):
            # Try common keys in order of preference
            for k in ["best", "val", "valid", "val_KL", "test_KL", "score", "loss"]:
                if k in metrics and metrics[k] is not None:
                    return float(metrics[k])
            # fallback: first numeric value
            for v in metrics.values():
                if isinstance(v, (int, float)):
                    return float(v)
            raise ValueError(f"Could not extract numeric score from dict metrics: {metrics}")

        # scalar
        if isinstance(metrics, (int, float)):
            return float(metrics)

        # tuple/list
        if isinstance(metrics, (list, tuple)) and len(metrics) > 0:
            # assume first element is the val metric
            if isinstance(metrics[0], (int, float)):
                return float(metrics[0])
            # or dict inside tuple
            if isinstance(metrics[0], dict):
                return self._extract_score(metrics[0])

        raise ValueError(f"Unsupported metrics format from run_model_main: {type(metrics)} / {metrics}")


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=str, default="H1")
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--study_name", type=str, default="gat_informer_arch_tuning")
    p.add_argument("--storage", type=str, default=None)  # e.g. sqlite:///optuna.db
    p.add_argument("--seed", type=int, default=42)

    # If you want, you can override decoding mode here
    p.add_argument("--decoding_mode", type=str, default="teacher-forced",
                   choices=["teacher-forced", "cost-aware-1", "cost-aware-2"])
    return p.parse_args()


if __name__ == "__main__":
    cli = _parse_cli()
    tuner = OptunaGATInformerTuner(
        cell=cli.cell,
        study_name=cli.study_name,
        storage=cli.storage,
        sampler_seed=cli.seed,
        decoding_mode=cli.decoding_mode,
    )
    tuner.run(n_trials=cli.n_trials, timeout=cli.timeout, n_jobs=cli.n_jobs)