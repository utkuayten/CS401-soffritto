import argparse
import optuna
from argparse import Namespace
from train_intra_cell import main as train_intra_cell_main
import os
import random
import pandas as pd
import gc
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning with Bag-of-Chromosomes subset + CV"
    )
    parser.add_argument(
        '--cell',
        type=str,
        required=True,
        help="Cell name (e.g. mESC or H1)"
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=10,
        help="Number of Optuna trials to run"
    )
    parser.add_argument(
        '--group_size',
        type=int,
        default=5,
        help="Consecutive group size for BOC sampling (default: 5)"
    )
    return parser.parse_args()


def get_all_chroms(cell: str):
    """
    Same logic as before:
    - mouse: 1..19
    - human: 1..22
    """
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 23))


def get_boc_subset(chroms_all, group_size, rng: random.Random):
    """
    Split chroms_all into consecutive groups of size group_size,
    pick 1 random chromosome from each group.
    """
    subset = []
    for i in range(0, len(chroms_all), group_size):
        group = chroms_all[i:i + group_size]
        if not group:
            continue
        chosen = rng.choice(group)
        subset.append(chosen)
    return sorted(subset)


def optuna_objective_boc(trial, cell: str, group_size: int = 5):
    chroms_all = get_all_chroms(cell)

    if len(chroms_all) == 0:
        raise ValueError("No chromosomes found for this cell.")

    # Per-trial RNG so that parallel Optuna runs are reproducible-ish
    rng = random.Random(trial.number)

    # 1) Pick a subset by BOC rule
    subset_chroms = get_boc_subset(chroms_all, group_size, rng)

    if len(subset_chroms) < 2:
        # Can't do leave-one-out CV with only 1 chromosome
        raise ValueError("BOC subset has fewer than 2 chromosomes; increase data or adjust group_size.")

    scores = []

    # ---- Tunable hyperparameters (sample ONCE per trial) ----
    e_layers = trial.suggest_int("e_layers", 1, 3)
    d_layers = trial.suggest_int("d_layers", 1, 3)
    d_model = trial.suggest_categorical("d_model", [256, 512, 1024])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    factor = trial.suggest_categorical("factor", [3, 5, 7])
    seq_len = trial.suggest_categorical("seq_len", [32, 64, 128])

    label_len = seq_len // 2  # int, derived from seq_len

    # 2) Cross-validation over the chosen subset
    for val_chrom in subset_chroms:
        train_chroms = [c for c in subset_chroms if c != val_chrom]

        trial_setting = (
            f"{cell}_BOCsubset_trial{trial.number}_val{val_chrom}"
        )

        args = Namespace(
            # Data / CV config
            cell=cell,
            train_chroms=train_chroms,
            val_chroms=[val_chrom],
            test_chroms=[val_chrom],
            setting=trial_setting,
            checkpoints=os.path.join("checkpoints", trial_setting),

            # Tunable hyperparameters
            e_layers=e_layers,
            d_layers=d_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            factor=factor,
            seq_len=seq_len,

            # Fixed
            batch_size=256,
            label_len=label_len,
            pred_len=1,
            enc_in=9,
            dec_in=16,
            c_out=16,
            activation='gelu',
            attn='prob',
            train_epochs=5,
            patience=3,
            lradj='type1',
            num_workers=4,
            use_multi_gpu=False,
            gpu=0,
            devices='0',
            selected_cols=[
                'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'
            ]
        )

        result = train_intra_cell_main(args)

        if result and isinstance(result, dict) and 'val_score' in result:
            scores.append(result['val_score'])

        # ---- MEMORY CLEANUP PER FOLD ----
        del result
        del args
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) Aggregate CV scores
    if not scores:
        # If no score returned (e.g. crashes), give a bad/neutral score
        return 0.0

    return sum(scores) / len(scores)


def main():
    args = parse_args()

    study = optuna.create_study(direction='minimize')  # e.g., KL divergence

    study.optimize(
        lambda trial: optuna_objective_boc(
            trial,
            cell=args.cell,
            group_size=args.group_size,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
    )

    print("\n[✓] Best Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save all trial results to CSV
    study_df = study.trials_dataframe()
    csv_path = f"optuna_BOC_{args.cell}.csv"
    study_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved Optuna BOC trials to: {csv_path}")


if __name__ == "__main__":
    main()