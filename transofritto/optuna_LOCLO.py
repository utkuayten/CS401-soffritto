#!/usr/bin/env python3
"""
optuna_LOCO_informer.py

Performs leave‐one‐cell‐line‐out hyperparameter optimization for Informer models
using Optuna. For each held‐out cell line, it concatenates the other cell lines'
CSV files into one training CSV, tunes hyperparameters, and finally retrains with
the best parameters, saving the best model and its hyperparameters.
"""

import os
import optuna
import pandas as pd
from types import SimpleNamespace
from run_model import run_model_main

class OptunaLOCOInformer:
    def __init__(self):
        # Define the five cell lines
        self.cell_lines = ['H1', 'H9', 'HCT116', 'mESC', 'mNPC']

        # For each held‐out cell, list the other four CSVs to train on
        self.train_features_files_dict = {
            'H1':     ["./data/H9_genomic.csv",
                       "./data/HCT116_genomic.csv",
                       "./data/mESC_genomic.csv",
                       "./data/mNPC_genomic.csv"],
            'H9':     ["./data/H1_genomic.csv",
                       "./data/HCT116_genomic.csv",
                       "./data/mESC_genomic.csv",
                       "./data/mNPC_genomic.csv"],
            'HCT116': ["./data/H1_genomic.csv",
                       "./data/H9_genomic.csv",
                       "./data/mESC_genomic.csv",
                       "./data/mNPC_genomic.csv"],
            'mESC':   ["./data/H1_genomic.csv",
                       "./data/H9_genomic.csv",
                       "./data/HCT116_genomic.csv",
                       "./data/mNPC_genomic.csv"],
            'mNPC':   ["./data/H1_genomic.csv",
                       "./data/H9_genomic.csv",
                       "./data/HCT116_genomic.csv",
                       "./data/mESC_genomic.csv"]
        }

        # Base directory for this script (where train_intra_cell.py / run_model.py live)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure directories exist:
        os.makedirs(os.path.join(self.base_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "temp_data"), exist_ok=True)

    def _merge_csvs(self, cell: str) -> str:
        """
        Concatenate the four training CSVs (for all cell lines except the held‐out 'cell')
        into a single temp CSV. Returns the temp CSV filepath.
        """
        csv_list = self.train_features_files_dict[cell]
        dfs = []
        for csv_path in csv_list:
            full_path = os.path.join(self.base_dir, csv_path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Cannot find training file: {full_path}")
            df = pd.read_csv(full_path)
            dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)

        temp_dir = os.path.join(self.base_dir, "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        merged_path = os.path.join(temp_dir, f"train_{cell}.csv")
        merged_df.to_csv(merged_path, index=False)
        return merged_path

    def _objective(self, trial, cell: str) -> float:
        """
        Optuna objective for a single held‐out cell line.
        - Merges the other four CSVs into one.
        - Samples hyperparameters.
        - Calls run_model_main(...) to train+validate.
        - Returns the validation score (to be minimized).
        """
        # 1) Merge training CSVs for this trial
        merged_train_csv = self._merge_csvs(cell)

        # 2) Build a namespace of arguments, matching run_model_main expectations
        args = SimpleNamespace()

        # Unique setting name so trials don't overwrite each other
        args.setting = f"{cell}_trial_{trial.number}"
        # The held‐out cell: used for naming/model output only
        args.cell = cell

        # Chromosome splits for the held‐out cell's own data (validation on held‐out cell)
        args.train_chroms = [1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
        args.val_chroms   = [6]

        # Sequence lengths
        args.seq_len   = 32
        args.label_len = 16
        args.pred_len  = 1

        # Architecture defaults (will be overridden by trial suggestions)
        args.enc_in  = 9
        args.dec_in  = 16
        args.c_out   = 16

        # Sample hyperparameters
        args.e_layers       = trial.suggest_int("e_layers", 1, 3)
        args.d_layers       = trial.suggest_int("d_layers", 1, 3)
        args.d_model        = trial.suggest_categorical("d_model", [256, 512, 1024])
        args.n_heads        = trial.suggest_categorical("n_heads", [2, 4, 8])
        args.d_ff           = trial.suggest_categorical("d_ff", [512, 1024, 2048])
        args.dropout        = trial.suggest_float("dropout", 0.05, 0.3)
        args.attn           = "prob"
        args.factor         = trial.suggest_categorical("factor", [3, 5, 7])
        args.activation     = "gelu"
        args.learning_rate  = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        args.train_epochs   = 10
        args.batch_size     = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        args.patience       = 3
        args.lradj          = "type1"
        args.weight_decay   = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        args.num_workers    = 5
        args.use_multi_gpu  = False
        args.gpu            = 0
        args.devices        = "0"
        args.output_attention = False
        args.distil = True
        args.model = 'informer'
        args.embed = "timeF"
        args.target = 'target_1'
        args.freq = 'w'
        args.mix = False

        # Data‐specific arguments
        args.data        = "custom"
        args.features    = "M"
        args.inverse     = False
        args.padding     = 0

        # Point run_model_main to the merged training CSV
        args.data_path   = merged_train_csv
        # The root path is not used by run_model_main for loading (since we provide full path),
        # but we set it to base_dir/data for consistency if needed internally.
        args.root_path   = os.path.join(self.base_dir, "data")
        args.checkpoints = os.path.join(self.base_dir, "checkpoints")

        # Device selection will happen inside run_model_main
        # Run training & validation; get back {"val_score": ...}
        metrics = run_model_main(args)
        return metrics["val_score"]

    def tune_cell(self, cell: str, n_trials: int = 30):
        """
        Perform an Optuna study for the given held‐out cell line.
        After optimization, retrain once on merged training CSV + best params.
        """
        print(f"[INFO] Starting hyperparameter tuning (LOCO) for held‐out cell: {cell}")
        study = optuna.create_study(direction="minimize", study_name=f"LOCO_{cell}")
        study.optimize(lambda trial: self._objective(trial, cell), n_trials=n_trials)

        best_val    = study.best_value
        best_params = study.best_params
        print(f"[INFO] [{cell}] Best validation score: {best_val:.6f}")
        print(f"[INFO] [{cell}] Best hyperparameters: {best_params}")

        # 1) Re‐merge the training CSVs for final training
        merged_train_csv = self._merge_csvs(cell)

        # 2) Build args with best params to retrain
        args = SimpleNamespace()
        args.setting      = f"{cell}_best"
        args.cell         = cell
        args.train_chroms = [1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22]
        args.val_chroms   = [6]

        args.seq_len   = 32
        args.label_len = 16
        args.pred_len  = 1

        args.enc_in  = 9
        args.dec_in  = 16
        args.c_out   = 16

        # Inject the best hyperparameters
        args.e_layers      = best_params["e_layers"]
        args.d_layers      = best_params["d_layers"]
        args.d_model       = best_params["d_model"]
        args.n_heads       = best_params["n_heads"]
        args.d_ff          = best_params["d_ff"]
        args.dropout       = best_params["dropout"]
        args.attn          = "prob"
        args.factor        = best_params["factor"]
        args.activation    = "gelu"
        args.learning_rate = best_params["learning_rate"]
        args.train_epochs  = 10
        args.batch_size    = best_params["batch_size"]
        args.patience      = 3
        args.lradj         = "type1"
        args.weight_decay  = best_params["weight_decay"]
        args.num_workers   = 5
        args.use_multi_gpu = False
        args.gpu           = 0
        args.devices       = "0"
        args.output_attention = False
        args.distil = True
        args.model = 'informer'
        args.embed = "timeF"
        args.target = 'target_1'
        args.freq = 'w'
        args.mix = False

        args.data        = "custom"
        args.features    = "M"
        args.inverse     = False
        args.padding     = 0

        args.data_path   = merged_train_csv
        args.root_path   = os.path.join(self.base_dir, "data")
        args.checkpoints = os.path.join(self.base_dir, "checkpoints")

        print(f"[INFO] Retraining {cell} with best hyperparameters ...")
        final_metrics = run_model_main(args)
        print(f"[INFO] [{cell}] Final training metrics: {final_metrics}")

    def run_all(self, n_trials_per_cell: int = 30):
        """
        Iterate over each cell line, performing LOCO tuning and final training.
        """
        for cell in self.cell_lines:
            self.tune_cell(cell, n_trials=n_trials_per_cell)


if __name__ == "__main__":
    tuner = OptunaLOCOInformer()
    tuner.run_all(n_trials_per_cell=30)