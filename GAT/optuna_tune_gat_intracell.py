# optuna_tune_gat_intracell.py
# Optuna tuner for the "NO COARSENING / NO POSITIONAL" GAT_intracell trainer you have.
#
# Assumes you have the class `GAT_intracell` available (import it from your script/module),
# and that utils.load_gat_intra_cell_line_train(...) works as before.

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import optuna


# -------------------------
# Reproducibility (optional)
# -------------------------
def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------
# Tuner
# -------------------------
@dataclass
class TuneConfig:
    features_file: str = "GAT/data/H1_features.npz"
    labels_file: str = "GAT/data/H1_labels.npz"
    test_chrom: str = "chr9"
    # train = all chr1..chr22 except chr6 and chr9
    drop_train_chroms: Tuple[str, ...] = ("chr6", "chr9")

    # training controls during tuning
    max_epochs: int = 120         # keep smaller for tuning
    patience: int = 20
    min_delta: float = 1e-5
    chroms_per_epoch: Optional[int] = 5  # or None to use all

    # fixed edges (you can also tune hop_list as categorical)
    hop_list: Tuple[int, ...] = (1, 10, 20)

    # device
    device: Optional[str] = None

    # objective type: "best_test_kl" from trainer
    direction: str = "minimize"


class OptunaGATTuner:
    """
    Wraps Optuna to tune your GAT_intracell hyperparameters.
    Expected: your GAT trainer exposes:
      - trainer.fit()
      - trainer.best_test_kl (float)  OR fit() returns a trainer with that attribute
    """
    def __init__(self, cfg: TuneConfig, trainer_cls, seed: int = 1337):
        self.cfg = cfg
        self.trainer_cls = trainer_cls
        self.seed = seed

        all_chroms = [f"chr{i}" for i in range(1, 23)]
        self.train_chroms = [c for c in all_chroms if c not in cfg.drop_train_chroms]

        seed_everything(seed)

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # Model capacity
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 48, 64])
        heads = trial.suggest_categorical("heads", [4, 8])
        layers = trial.suggest_int("layers", 1, 4)

        # Regularization
        dropout = trial.suggest_float("dropout", 0.00, 0.30)
        widen = trial.suggest_categorical("widen", [2, 4, 6])

        # Optim
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

        # Training stability
        grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)

        # Optional: tune hop_list too (uncomment if you want)
        # hop_list = trial.suggest_categorical(
        #     "hop_list",
        #     [
        #         (1, 2, 4, 8),
        #         (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        #         (1, 5, 10, 15, 20),
        #         (1, 10, 20),
        #     ],
        # )

        return dict(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            widen=widen,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            hop_list=self.cfg.hop_list,  # or hop_list if tuning it
        )

    def objective(self, trial: optuna.Trial) -> float:
        seed_everything(self.seed + trial.number)

        p = self._suggest_params(trial)

        # Create trainer for this trial
        trainer = self.trainer_cls(
            features_file=self.cfg.features_file,
            labels_file=self.cfg.labels_file,
            train_chromosomes=self.train_chroms,
            test_chromosome=self.cfg.test_chrom,

            hop_list=p["hop_list"],

            hidden_dim=p["hidden_dim"],
            heads=p["heads"],
            layers=p["layers"],
            dropout=p["dropout"],
            widen=p["widen"],

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

        # Score: use best test KL from training
        score = float(getattr(trainer, "best_test_kl", math.inf))

        # Report to Optuna (so it can prune if you later add a pruner)
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def run(
            self,
            n_trials: int = 50,
            study_name: str = "gat_intracell_tune",
            storage: Optional[str] = None,
            load_if_exists: bool = True,
            sampler: Optional[optuna.samplers.BaseSampler] = None,
            pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            direction=self.cfg.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists,
        )
        study.optimize(self.objective, n_trials=n_trials, gc_after_trial=True)

        print("\nBest value:", study.best_value)
        print("Best params:", study.best_params)

        return study

    def train_best_and_save_predictions(
            self,
            best_params: Dict[str, Any],
            out_path: str = "GAT/predictions/H1_predictions.npz",
            final_epochs: int = 200,
            final_patience: int = 30,
    ) -> None:
        """
        Re-train using best params (optionally with more epochs),
        then save chr9 predictions to NPZ.
        """
        seed_everything(self.seed)

        # If hop_list is not in best_params (because you kept it fixed), use cfg.hop_list
        hop_list = best_params.get("hop_list", self.cfg.hop_list)

        trainer = self.trainer_cls(
            features_file=self.cfg.features_file,
            labels_file=self.cfg.labels_file,
            train_chromosomes=self.train_chroms,
            test_chromosome=self.cfg.test_chrom,
            hop_list=hop_list,

            hidden_dim=int(best_params["hidden_dim"]),
            heads=int(best_params["heads"]),
            layers=int(best_params["layers"]),
            dropout=float(best_params["dropout"]),
            widen=int(best_params["widen"]),

            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
            epochs=final_epochs,
            grad_clip=float(best_params.get("grad_clip", 1.0)),

            patience=final_patience,
            min_delta=self.cfg.min_delta,
            chroms_per_epoch=self.cfg.chroms_per_epoch,

            device=self.cfg.device,
        )

        trainer.fit()
        probs = trainer.predict_test_probs()  # [N, C] for chr9 (NO COARSENING)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, test_chromosome=self.cfg.test_chrom, probs=probs)
        print("saved:", out_path, "shape:", probs.shape)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Import your trainer class from your script:
    # from gat_intracell_keep_testshape import GAT_intracell
    from train_intra_cell_line import GAT_intracell  # <-- change if your filename differs

    cfg = TuneConfig(
        features_file="GAT/data/H1_features.npz",
        labels_file="GAT/data/H1_labels.npz",
        test_chrom="chr9",
        hop_list=(1, 10, 20),
        chroms_per_epoch=5,
        max_epochs=120,
        patience=20,
    )

    tuner = OptunaGATTuner(cfg, trainer_cls=GAT_intracell, seed=1337)

    # If you want persistent storage:
    # study = tuner.run(n_trials=50, storage="sqlite:///optuna_gat.db")

    study = tuner.run(n_trials=50)

    # Retrain best and save predictions
    tuner.train_best_and_save_predictions(
        best_params=study.best_params,
        out_path="GAT/predictions/H1_predictions.npz",
        final_epochs=200,
        final_patience=30,
    )