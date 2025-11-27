#!/usr/bin/env python3
"""
CV_all_iterations.py

Runs “leave‐one‐chromosome‐out” cross‐validation on each cell line.
For each cell:
  - Fix a single “test” chromosome (default = 9).
  - On the remaining chromosomes, repeatedly hold out one
    as validation and train on the rest.
  - Collect each fold’s best validation loss, then average.

Usage:
    python CV_all_iterations.py

Requirements:
    - The transofritto/informer/exp/exp_informer.py must define Exp_Informer
      with a .train(setting) method that returns (model, best_val_loss).
    - For each cell (e.g. “H1”), you must have:
          best_model/H1_CV_informer/H1_val_6_hyperparameters.json
      which contains all Informer hyperparameters (as a JSON dict).
    - Data files: transofritto/data/{cell}_genomic.csv must exist.
"""

import os
import sys
import json
import numpy as np
import torch

# ─── Make sure our project root is on PYTHONPATH ─────────────────────────────
THIS_DIR    = os.path.dirname(__file__)
PROJECT_ROOT= os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer

# ─── Utility to load hyperparameters from the JSON file ───────────────────────
def load_hyperparams(json_path):
    """
    json_path: Path to a JSON file containing Informer hyperparameters.
    Returns: a Python dict.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

# ─── Build a Namespace‐like object from hyperparams + constants ──────────────
def build_args_for_fold(cell, hp, train_chroms, val_chrom, test_chrom, device):
    """
    Construct an args‐object for Exp_Informer for one CV fold.

    cell: str, e.g. "H1"
    hp:   dict of hyperparameters loaded from JSON
    train_chroms: list of ints to train on
    val_chrom:    int for this fold’s validation chromosome
    test_chrom:   int for the held‐out test chromosome
    device:       torch.device (e.g. torch.device("mps") or "cuda")

    Returns: args object with all needed fields for Exp_Informer.
    """
    # 1) Start with an empty simple object
    args = type("A", (), {})()

    # 2) Copy over everything from the JSON hyperparams
    #    (expects keys like seq_len, label_len, pred_len, d_model, e_layers, etc.)
    for k, v in hp.items():
        setattr(args, k, v)

    # 3) Override / set cell‐specific fields:
    #    Model ID (used for saving checkpoints), data paths, and chromosome splits.
    args.model_id   = f"{cell}_CV_informer"
    args.model      = "informer"
    args.data       = "custom"
    args.features   = "M"
    args.target     = "target_1"
    args.freq       = hp.get("freq", "w")
    args.root_path  = "transofritto/data"
    args.data_path  = f"{cell}_genomic.csv"
    args.checkpoints= f"best_model/{cell}_CV_informer"
    args.enc_in     = 9
    args.dec_in     = 16
    args.c_out      = 16

    # Chromosomes to use for training, validation, and testing
    args.train_chroms = train_chroms
    args.val_chroms   = [val_chrom]
    args.test_chroms  = [test_chrom]

    # We also set args.chromosome = val_chrom, so that when Exp_Informer
    # calls data_provider(flag="val"), it knows which chromosome to load.
    args.chromosome = val_chrom

    # 4) GPU / device settings
    args.use_gpu      = True
    args.use_multi_gpu= False
    args.devices      = "0"
    args.gpu          = 0
    args.device       = device
    args.batch_size   = 64          # adjust if desired
    args.num_workers  = 4           # adjust to your CPU cores

    # 5) Informer‐specific constants (must match what was used in training JSON)
    #    If the JSON already includes distil, attn, embed, etc., no need to override.
    #    Otherwise, set them here. For safety, we ensure all required fields exist:
    if not hasattr(args, "distil"):
        args.distil = True
    if not hasattr(args, "attn"):
        args.attn = "prob"
    if not hasattr(args, "embed"):
        args.embed = "timeF"
    if not hasattr(args, "activation"):
        args.activation = "gelu"
    if not hasattr(args, "padding"):
        args.padding = 0
    if not hasattr(args, "use_amp"):
        args.use_amp = False
    if not hasattr(args, "inverse"):
        args.inverse = False
    if not hasattr(args, "output_attention"):
        args.output_attention = False

    return args

# ─── Load a pretrained Informer checkpoint for inference ──────────────────────
def load_checkpoint(exp, path, device):
    """
    Load a .pth checkpoint (which may be a folder or a file) into exp.model.
    """
    ckpt = os.path.join(path, "checkpoint.pth") if os.path.isdir(path) else path
    data = torch.load(ckpt, map_location=device)
    # Some checkpoints nest under "state_dict"
    sd = data.get("state_dict", data)
    # Clean out any DataParallel prefixes
    clean = {k.replace("module.", ""): v for k, v in sd.items()}
    exp.model.load_state_dict(clean)

# ─── Run one CV fold: train on train_chroms, validate on val_chrom ─────────────
def train_one_fold(cell, json_hp_path, train_chroms, val_chrom, test_chrom, device):
    """
    - Builds args for this fold
    - Instantiates Exp_Informer
    - Calls exp.train(setting)
    - Returns best validation loss (float)
    """
    # 1) Load hyperparameters from JSON
    hp = load_hyperparams(json_hp_path)

    # 2) Build args with correct splits
    args = build_args_for_fold(cell, hp, train_chroms, val_chrom, test_chrom, device)

    # 3) Instantiate and train
    exp = Exp_Informer(args)
    exp.device = args.device
    exp.model.to(args.device)

    # Each fold can have a unique “setting” name for saving checkpoints:
    setting_name = f"{cell}_val_chrom{val_chrom}"
    # Make sure the checkpoint directory exists
    fold_ckpt_dir = os.path.join(args.checkpoints, setting_name)
    os.makedirs(fold_ckpt_dir, exist_ok=True)

    # Call train() → returns (best_model, best_validation_loss)
    best_model, best_val_loss = exp.train(setting_name)

    # Optionally delete best_model from memory to free GPU RAM
    del best_model
    torch.cuda.empty_cache()

    return best_val_loss

def get_available_chroms(cell):
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 22))
# ─── For a given cell line, run leave‐one‐chromosome‐out CV (except test chromosome) ─
def run_cv_for_cell(cell, json_hp_path, test_chrom=9, device=None):
    """
    For the given `cell`, hold out `test_chrom` as PURE test set,
    then on the remaining chromosomes do:
       for each val_chrom in (all_chroms \ {test_chrom}):
          - train_chroms = (all_chroms \ {test_chrom, val_chrom})
          - call train_one_fold(...)
          - collect best_val_loss
    Finally average all best_val_loss values and return that average.
    """
    # 1) Device selection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # 2) Prepare chromosome lists
    all_chroms = get_available_chroms(cell)
    if test_chrom not in all_chroms:
        raise ValueError(f"test_chrom={test_chrom} not in 1..22")
    # Remove test_chrom from CV pool
    cv_pool = [c for c in all_chroms if c != test_chrom]

    # 3) Iterate through each validation chromosome
    val_losses = []
    print(f"\n--- Running CV for cell {cell}, test_chrom={test_chrom} ---")
    for val_chrom in cv_pool:
        # training chromosomes = all in cv_pool except val_chrom
        train_chroms = [c for c in cv_pool if c != val_chrom]

        print(f"Fold: validation on chr{val_chrom}, train on {train_chroms}")
        best_val_loss = train_one_fold(
            cell=cell,
            json_hp_path=json_hp_path,
            train_chroms=train_chroms,
            val_chrom=val_chrom,
            test_chrom=test_chrom,
            device=device
        )
        print(f"  → Best val loss on chr{val_chrom}: {best_val_loss:.6f}\n")
        val_losses.append(best_val_loss)

    # 4) Compute and return the average validation loss
    avg_val_loss = float(np.mean(val_losses))
    print(f"*** Cell {cell}: average CV validation loss = {avg_val_loss:.6f} ***\n")
    return avg_val_loss, val_losses

# ─── Main: run CV on each cell line and save results ────────────────────────────
if __name__ == "__main__":
    # List of all cell lines to run CV on
    cell_lines = ["H9"]

    # By default, hold out chromosome 9 as the pure test set for every cell.
    # Change test_chrom if you want a different chromosome.
    test_chrom = 9

    # Device: prefer GPU, else MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Where to store the CV summary
    summary_out = os.path.join(PROJECT_ROOT, "CV_summary.txt")
    with open(summary_out, "w") as fout:
        fout.write("Cell\tAvg_CV_Val_Loss\tAll_Fold_Losses\n")

    # 1) Loop over each cell
    for cell in cell_lines:
        # Path to that cell’s hyperparameter JSON
        json_hp_path = (
            f"{PROJECT_ROOT}/transofritto/best_model/{cell}_CV_informer/"
            f"{cell}_val_6_hyperparameters.json"
        )
        if not os.path.exists(json_hp_path):
            raise FileNotFoundError(f"Cannot find hyperparam JSON: {json_hp_path}")

        # 2) Run leave‐one‐chrom CV for that cell
        avg_loss, all_losses = run_cv_for_cell(
            cell=cell,
            json_hp_path=json_hp_path,
            test_chrom=test_chrom,
            device=device
        )

        # 3) Append results to summary file
        with open(summary_out, "a") as fout:
            fout.write(f"{cell}\t{avg_loss:.6f}\t{all_losses}\n")

    print(f"\nAll done! CV summary written to:\n    {summary_out}\n")