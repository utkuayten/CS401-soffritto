# =========================
# optuna_tune_gat_intracell.py
# (NO chunk_len anywhere)
# ✅ Generic cell_line + generic save paths
# ✅ No pandas dependency for CSV
# =========================
from __future__ import annotations

import os
import math
import random
import gc
import json
import csv
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

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
    # ✅ ONE switch
    cell_line: str = "H1"
    run_tag: str = ""  # optional suffix to avoid overwriting, e.g. "run1"

    # folders
    data_dir: str = "GAT/data"
    out_dir: str = "GAT"

    # auto (can override if you want)
    features_file: Optional[str] = None
    labels_file: Optional[str] = None

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

    # auto (saved under trials/<run_name>/trials.csv)
    trials_csv: Optional[str] = None

    def __post_init__(self) -> None:
        if self.features_file is None:
            self.features_file = os.path.join(self.data_dir, f"{self.cell_line}_features.npz")
        if self.labels_file is None:
            self.labels_file = os.path.join(self.data_dir, f"{self.cell_line}_labels.npz")

        if self.trials_csv is None:
            self.trials_csv = os.path.join(self.out_dir, "trials", self.run_name, "trials.csv")

    @property
    def run_name(self) -> str:
        return self.cell_line if not self.run_tag else f"{self.cell_line}_{self.run_tag}"

    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.out_dir, "checkpoints", self.run_name, "best.pth")

    @property
    def best_pred_path(self) -> str:
        return os.path.join(self.out_dir, "predictions", self.run_name, f"{self.cell_line}_predictions.npz")

    @property
    def best_params_path(self) -> str:
        return os.path.join(self.out_dir, "trials", self.run_name, "best_params.json")


class OptunaGATTuner:
    def __init__(self, cfg: TuneConfig, trainer_cls, seed: int = 1337):
        self.cfg = cfg
        self.trainer_cls = trainer_cls
        self.seed = seed

        # mouse: chr1..chr19, human: chr1..chr22
        max_chr = 19 if cfg.cell_line.lower().startswith("m") else 22
        all_chroms = [f"chr{i}" for i in range(1, max_chr + 1)]

        # drop only those that actually exist
        drop = tuple(c for c in cfg.drop_train_chroms if c in all_chroms)
        self.train_chroms = [c for c in all_chroms if c not in drop and c != cfg.test_chrom]

        seed_everything(seed)

        # Track best within the Optuna loop (saved during objective)
        self.best_value: float = float("inf")
        self.best_trial_number: int = -1

        # ✅ generic paths (per cell_line/run_tag)
        self.best_ckpt_path: str = self.cfg.best_ckpt_path
        self.best_pred_path: str = self.cfg.best_pred_path
        self.best_params_path: str = self.cfg.best_params_path

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

        # --- hop list id (kept for compatibility; only useful if trainer uses it internally) ---
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

    def _extract_model(self, trainer):
        model = None
        for attr in ("model", "net", "module"):
            if hasattr(trainer, attr):
                cand = getattr(trainer, attr)
                if cand is not None and hasattr(cand, "state_dict"):
                    model = cand
                    break
        if model is None and hasattr(trainer, "state_dict"):
            try:
                _ = trainer.state_dict()
                model = trainer
            except Exception:
                model = None
        return model

    def _maybe_save_best(self, score: float, trial_number: int, params: Dict[str, Any], trainer) -> None:
        if score >= self.best_value:
            return

        self.best_value = score
        self.best_trial_number = int(trial_number)

        # save best params json (updated as soon as best improves)
        os.makedirs(os.path.dirname(self.best_params_path), exist_ok=True)
        with open(self.best_params_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": self.cfg.run_name,
                    "cell_line": self.cfg.cell_line,
                    "best_value": float(self.best_value),
                    "best_trial_number": int(self.best_trial_number),
                    "best_params": dict(params),
                },
                f,
                indent=2,
                sort_keys=True,
            )

        # save best checkpoint (overwrite)
        model = self._extract_model(trainer)
        if model is not None:
            os.makedirs(os.path.dirname(self.best_ckpt_path), exist_ok=True)
            ckpt = {
                "state_dict": model.state_dict(),
                "hparams": dict(params),
                "best_value": float(self.best_value),
                "best_trial_number": int(self.best_trial_number),
                "cell_line": self.cfg.cell_line,
                "run_name": self.cfg.run_name,
                "test_chromosome": self.cfg.test_chrom,
                "features_file": self.cfg.features_file,
                "labels_file": self.cfg.labels_file,
            }
            if hasattr(trainer, "best_test_kl"):
                ckpt["best_test_kl"] = float(trainer.best_test_kl)
            torch.save(ckpt, self.best_ckpt_path)
            print(
                f"[BEST] saved checkpoint: {self.best_ckpt_path} (trial={self.best_trial_number}, value={self.best_value:.6f})")
        else:
            print("[BEST][WARN] Could not find model to save best.pth (trainer.model/net/module missing).")

        # save best predictions (overwrite)
        if hasattr(trainer, "predict_test_probs"):
            try:
                probs = trainer.predict_test_probs()
                os.makedirs(os.path.dirname(self.best_pred_path), exist_ok=True)
                np.savez_compressed(
                    self.best_pred_path,
                    cell_line=self.cfg.cell_line,
                    run_name=self.cfg.run_name,
                    test_chromosome=self.cfg.test_chrom,
                    probs=probs,
                )
                print(f"[BEST] saved predictions: {self.best_pred_path} shape={getattr(probs, 'shape', None)}")
            except Exception as e:
                print(f"[BEST][WARN] Failed to save predictions: {e}")
        else:
            print("[BEST][WARN] trainer has no predict_test_probs(); skipping prediction save.")

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

            # ✅ Save best model/preds during the Optuna loop (no final retrain needed)
            self._maybe_save_best(score=score, trial_number=trial.number, params=p, trainer=trainer)

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

    @staticmethod
    def _save_trials_csv(csv_path: str, trials: List[optuna.trial.FrozenTrial]) -> None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # simple + robust schema (no pandas)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["number", "state", "value", "params_json"])
            for t in trials:
                w.writerow([
                    int(t.number),
                    str(t.state),
                    "" if t.value is None else float(t.value),
                    json.dumps(t.params, sort_keys=True),
                ])

    def run(
            self,
            n_trials: int = 10,
            study_name: Optional[str] = None,
            storage: Optional[str] = None,
            load_if_exists: bool = True,
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

        if study_name is None:
            study_name = f"gat_intracell_tune_{self.cfg.run_name}_fullchrom"

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

        # Save CSV
        self._save_trials_csv(self.cfg.trials_csv, study.trials)
        print("saved trials csv:", self.cfg.trials_csv)

        # (best_params.json / best.pth / best predictions are already written during optimization)
        return study

    def train_best_and_save_predictions(
            self,
            best_params: Dict[str, Any],
            out_path: Optional[str] = None,
            final_epochs: int = 200,
            final_patience: int = 30,
            final_device: str = "cuda:0",
    ) -> None:
        seed_everything(self.seed)
        if "cuda" in final_device and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if out_path is None:
            out_path = self.cfg.best_pred_path

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

        # ✅ Save best.pth (generic path)
        ckpt_path = self.cfg.best_ckpt_path
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        model = self._extract_model(trainer)
        if model is None:
            raise AttributeError(
                "Could not find a model to save. Expected trainer.model/net/module (with state_dict) "
                "or trainer.state_dict()."
            )

        ckpt = {
            "state_dict": model.state_dict(),
            "hparams": dict(best_params),
            "cell_line": self.cfg.cell_line,
            "run_name": self.cfg.run_name,
            "test_chromosome": self.cfg.test_chrom,
            "features_file": self.cfg.features_file,
            "labels_file": self.cfg.labels_file,
        }
        if hasattr(trainer, "best_test_kl"):
            ckpt["best_test_kl"] = float(trainer.best_test_kl)

        torch.save(ckpt, ckpt_path)
        print("saved best checkpoint:", ckpt_path)

        probs = trainer.predict_test_probs()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            cell_line=self.cfg.cell_line,
            run_name=self.cfg.run_name,
            test_chromosome=self.cfg.test_chrom,
            probs=probs,
        )
        print("saved:", out_path, "shape:", probs.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_line", type=str, default="H1")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--test_chrom", type=str, default="chr9")
    parser.add_argument("--chroms_per_epoch", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--no_load_if_exists", action="store_true")
    args = parser.parse_args()

    from train_intra_cell import GAT_intracell

    cfg = TuneConfig(
        cell_line=args.cell_line,
        run_tag=args.run_tag,
        test_chrom=args.test_chrom,
        chroms_per_epoch=args.chroms_per_epoch,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        n_jobs=1,
    )

    tuner = OptunaGATTuner(cfg, trainer_cls=GAT_intracell, seed=1337)
    tuner.run(
        n_trials=args.n_trials,
        storage=args.storage,
        load_if_exists=(not args.no_load_if_exists),
    )

    # best.pth + best_params.json + best predictions are saved during the Optuna loop


if __name__ == "__main__":
    main()