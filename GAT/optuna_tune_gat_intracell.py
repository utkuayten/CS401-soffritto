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
from datetime import datetime


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

    # local edges INSIDE CHUNKS
    hop_list: Tuple[int, ...] = (1, 2, 4)

    # streaming chunk length (this is Soffritto "batch_size" concept)
    # tune this; bigger => more memory, more context per step
    chunk_len_choices: Tuple[int, ...] = (256, 512, 1024)

    # ✅ GPU
    device: str = "cuda:0"

    direction: str = "minimize"

    # ✅ single GPU => must be 1
    n_jobs: int = 1


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
        # --- GAT encoder (keep small; attention is expensive) ---
        gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4])
        gat_hidden = trial.suggest_categorical("gat_hidden", [4, 8, 16])

        # --- LSTM (this is what matters most vs Soffritto) ---
        num_hiddens = trial.suggest_categorical("num_hiddens", [16, 32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)

        # --- streaming ---
        chunk_len = 2048 * 2

        # --- hop list (local receptive field inside chunk) ---
        hop_space = {
            # "h1":  (1,),
            # "h12": (1, 2),
            # "h124": (1, 2, 4),
            # "h1248": (1, 2, 4, 8),
            # "h12416": (1, 2, 4, 16),
            # "h124816": (1, 2, 4, 8, 16),
            "h12345": (1, 2, 3, 4, 5,),
            "h124": (1, 2, 4,),
            "h12481632": (1, 2, 4, 8, 16, 32),
            # "h24": (2, 4),
            # "h48": (4, 8),
        }
        hop_id = trial.suggest_categorical("hop_id", list(hop_space.keys()))
        hop_list = hop_space[hop_id]

        # --- regular training ---
        dropout = trial.suggest_float("dropout", 0.0, 0.30)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        grad_clip = 0  # trial.suggest_float("grad_clip", 0.5, 2.0)

        return dict(
            gat_hidden=gat_hidden,
            gat_heads=gat_heads,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            chunk_len=chunk_len,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,

            # ✅ now tunable
            hop_id=hop_id,
            hop_list=hop_list,
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

                hop_list=p["hop_list"],

                # ✅ new trainer args
                gat_hidden=p["gat_hidden"],
                gat_heads=p["gat_heads"],
                num_hiddens=p["num_hiddens"],
                num_layers=p["num_layers"],
                chunk_len=2048,

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
            # Mark this trial as bad instead of crashing the whole study
            if "cuda" in self.cfg.device:
                torch.cuda.empty_cache()
            return float("inf")

        finally:
            gc.collect()
            if "cuda" in self.cfg.device and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(
            self,
            n_trials: int = 5,
            study_name: str = "gat_intracell_tune",
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
            n_jobs=self.cfg.n_jobs,  # 1 on GPU
        )

        print("\nBest value:", study.best_value)
        print("Best params:", study.best_params)

        # ---------------------------
        # ✅ All trials in SAME CSV
        # ---------------------------
        os.makedirs("GAT/trials", exist_ok=True)
        out_csv = "GAT/trials/trials.csv"

        df_new = study.trials_dataframe(
            attrs=("number", "state", "value", "params", "user_attrs", "system_attrs")
        )

        if os.path.exists(out_csv):
            df_old = pd.read_csv(out_csv)
            # avoid duplicates by trial number (keep old if repeated)
            df_all = (
                pd.concat([df_old, df_new], ignore_index=True)
                .drop_duplicates(subset=["number"], keep="last")
                .sort_values("number")
            )
        else:
            df_all = df_new.sort_values("number")

        df_all.to_csv(out_csv, index=False)
        print("saved trials csv:", out_csv)

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

            hop_list=self.cfg.hop_list,

            gat_hidden=int(best_params["gat_hidden"]),
            gat_heads=int(best_params["gat_heads"]),
            num_hiddens=int(best_params["num_hiddens"]),
            num_layers=int(best_params["num_layers"]),
            chunk_len=int(2048),

            dropout=float(best_params["dropout"]),

            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
            epochs=final_epochs,
            grad_clip=float(0),

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
    # ✅ IMPORTANT: import the streaming trainer
    from train_intra_cell_line import GAT_intracell

    cfg = TuneConfig(
        hop_list=(1, 2, 4, 8),
        chroms_per_epoch=20,   # tune speed vs stability
        max_epochs=300,
        patience=20,
        device="cuda:0",
        n_jobs=1,
    )

    tuner = OptunaGATTuner(cfg, trainer_cls=GAT_intracell, seed=1337)
    study = tuner.run(n_trials=10)

    tuner.train_best_and_save_predictions(
        best_params=study.best_params,
        out_path="GAT/predictions/H1_predictions.npz",
        final_epochs=10,
        final_patience=30,
        final_device="cuda:0",
    )