# =========================
# optuna_tune_gat_intracell.py
# (NO chunk_len anywhere)
# =========================
from __future__ import annotations

import os
import math
import random
import gc
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import optuna


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TuneConfig:
    features_file: str = "GAT/data/H1_features.npz"
    labels_file: str = "GAT/data/H1_labels.npz"
    test_chrom: str = "chr9"
    drop_train_chroms: Tuple[str, ...] = ("chr6", "chr9")

    max_epochs: int = 120
    patience: int = 20
    min_delta: float = 1e-5

    # how many chromosomes per epoch (None => all)
    chroms_per_epoch: Optional[int] = None


    # ✅ GPU
    device: str = "cuda:0"

    direction: str = "minimize"

    # single GPU => keep 1
    n_jobs: int = 1

    # results
    trials_csv: str = "GAT/trials/trials.csv"


class OptunaGATTuner:
    def __init__(self, cfg: TuneConfig, trainer_cls, seed: int = 1337):
        self.cfg = cfg
        self.trainer_cls = trainer_cls
        self.seed = seed

        all_chroms = [f"chr{i}" for i in range(1, 23)]
        self.train_chroms = [c for c in all_chroms if c not in cfg.drop_train_chroms]

        seed_everything(seed)

        # Force single job on CUDA to avoid OOM / contention
        if "cuda" in self.cfg.device:
            if not torch.cuda.is_available():
                raise RuntimeError("cfg.device is CUDA but torch.cuda.is_available() is False.")
            if self.cfg.n_jobs != 1:
                print("[WARN] Single GPU => forcing n_jobs=1")
                self.cfg.n_jobs = 1
            torch.cuda.set_device(int(self.cfg.device.split(":")[1]) if ":" in self.cfg.device else 0)

        # avoid CPU oversubscription
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # --- GAT encoder (attention is expensive) ---
        gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4])
        gat_hidden = trial.suggest_categorical("gat_hidden", [4, 8, 16])

        # --- LSTM ---
        num_hiddens = trial.suggest_categorical("num_hiddens", [16, 32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)

        # --- hop list (full-chrom graph) ---
        hop_space = {
            "h124": (1, 2, 4),
            "h1248": (1, 2, 4, 8),
            "h12481632": (1, 2, 4, 8, 16, 32),
        }
        hop_id = trial.suggest_categorical("hop_id", list(hop_space.keys()))

        # --- training ---
        dropout = trial.suggest_float("dropout", 0.0, 0.15)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        grad_clip = 0.0  # keep 0 unless you want to tune it

        return dict(
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            hop_id=hop_id,
        )

    def objective(self, trial: optuna.Trial) -> float:
        seed_everything(self.seed + trial.number)

        # reduce fragmentation between trials
        if "cuda" in self.cfg.device:
            torch.cuda.empty_cache()

        p = self._suggest_params(trial)

        try:
            trainer = self.trainer_cls(
                features_file=self.cfg.features_file,
                labels_file=self.cfg.labels_file,
                train_chromosomes=self.train_chroms,
                test_chromosome=self.cfg.test_chrom,


                gat_hidden=p["gat_hidden"],
                gat_heads=p["gat_heads"],
                num_hiddens=p["num_hiddens"],
                num_layers=p["num_layers"],
                dropout=p["dropout"],

                lr=p["lr"],
                weight_decay=p["weight_decay"],
                epochs=self.cfg.max_epochs,
                grad_clip=p["grad_clip"],

                patience=self.cfg.patience,
                min_delta=self.cfg.min_delta,
                chroms_per_epoch=self.cfg.chroms_per_epoch,

                device=self.cfg.device,
            )

            trainer.fit()
            score = float(getattr(trainer, "best_test_kl", math.inf))

            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return score

        except torch.cuda.OutOfMemoryError:
            if "cuda" in self.cfg.device:
                torch.cuda.empty_cache()
            return float("inf")

        finally:
            gc.collect()
            if "cuda" in self.cfg.device and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(
        self,
        n_trials: int = 10,
        study_name: str = "gat_intracell_tune_fullchrom",
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            direction=self.cfg.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            gc_after_trial=True,
            n_jobs=self.cfg.n_jobs,  # keep 1 for single GPU
        )

        print("\nBest value:", study.best_value)
        print("Best params:", study.best_params)

        # Save CSV (no pandas dependency)
        os.makedirs(os.path.dirname(self.cfg.trials_csv), exist_ok=True)
        df = study.trials_dataframe(attrs=("number", "state", "value", "params"))
        df.to_csv(self.cfg.trials_csv, index=False)
        print("saved trials csv:", self.cfg.trials_csv)

        return study

    def train_best_and_save_predictions(
        self,
        best_params: Dict[str, Any],
        out_path: str = "GAT/predictions/H1_predictions.npz",
        final_epochs: int = 200,
        final_patience: int = 30,
        final_device: str = "cuda:0",
    ) -> None:
        seed_everything(self.seed)
        if "cuda" in final_device and torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer = self.trainer_cls(
            features_file=self.cfg.features_file,
            labels_file=self.cfg.labels_file,
            train_chromosomes=self.train_chroms,
            test_chromosome=self.cfg.test_chrom,


            gat_hidden=int(best_params["gat_hidden"]),
            gat_heads=int(best_params["gat_heads"]),
            num_hiddens=int(best_params["num_hiddens"]),
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),

            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
            epochs=final_epochs,
            grad_clip=float(best_params.get("grad_clip", 0.0)),

            patience=final_patience,
            min_delta=self.cfg.min_delta,
            chroms_per_epoch=self.cfg.chroms_per_epoch,

            device=final_device,
        )

        trainer.fit()
        probs = trainer.predict_test_probs()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, test_chromosome=self.cfg.test_chrom, probs=probs)
        print("saved:", out_path, "shape:", probs.shape)


if __name__ == "__main__":
    from train_intra_cell import GAT_intracell

    cfg = TuneConfig(
        chroms_per_epoch=20,
        max_epochs=500,
        patience=20,
        device="cuda:0",
        n_jobs=1,
    )

    tuner = OptunaGATTuner(cfg, trainer_cls=GAT_intracell, seed=1337)
    study = tuner.run(n_trials=20)

    tuner.train_best_and_save_predictions(
        best_params=study.best_params,
        out_path="GAT/predictions/H1_predictions.npz",
        final_epochs=10,
        final_patience=30,
        final_device="cuda:0",
    )