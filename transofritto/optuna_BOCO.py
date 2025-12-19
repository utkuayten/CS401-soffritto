import argparse
import optuna
from argparse import Namespace
from train_intra_cell import main as train_intra_cell_main
import os
import random
import gc
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning with Bag-of-Chromosomes-OUT (BOCO) + CV"
    )
    parser.add_argument('--cell', type=str, required=True, help="Cell name (e.g. mESC or H1)")
    parser.add_argument('--n_trials', type=int, default=10, help="Number of Optuna trials to run")
    parser.add_argument('--group_size', type=int, default=5, help="Consecutive group size for BOC sampling")
    parser.add_argument('--n_jobs', type=int, default=1, help="Parallel Optuna jobs (recommend 1 on MPS)")
    return parser.parse_args()


def get_all_chroms(cell: str):
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 23))


def get_boc_subset(chroms_all, group_size, rng: random.Random):
    subset = []
    for i in range(0, len(chroms_all), group_size):
        group = chroms_all[i:i + group_size]
        if group:
            subset.append(rng.choice(group))
    return sorted(subset)


def optuna_objective_boco(trial, cell: str, group_size: int = 5):
    chroms_all = get_all_chroms(cell)
    if not chroms_all:
        raise ValueError("No chromosomes found for this cell.")

    rng = random.Random(trial.number)

    # 1) BOCO bag (held-out set)
    bag_chroms = get_boc_subset(chroms_all, group_size, rng)
    if not bag_chroms:
        raise ValueError("BOCO bag is empty; check group_size or chromosome list.")

    train_chroms_all = [c for c in chroms_all if c not in bag_chroms]
    if not train_chroms_all:
        raise ValueError("All chromosomes ended up in the bag; no training data remains.")

    scores = []

    # -------------------------
    # Tunable hyperparameters
    # -------------------------
    # GAT-specific
    gat_window = trial.suggest_categorical("gat_window", [2, 4, 8, 12, 16])
    gat_alpha = trial.suggest_categorical("gat_alpha", [0.1, 0.2, 0.3])

    e_layers = trial.suggest_int("e_layers", 1, 4)
    d_layers = trial.suggest_int("d_layers", 1, 4)

    d_model = trial.suggest_categorical("d_model", [256, 512, 1024])
    valid_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048])
    dropout = trial.suggest_float("dropout", 0.05, 0.15)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    factor = trial.suggest_categorical("factor", [3, 5, 7])

    seq_len = 32
    label_len = seq_len // 2

    # -------------------------
    # Fixed (match your working run)
    # -------------------------
    enc_in = 9
    dec_in = 16
    c_out = 16
    attn = "gat"

    # 2) Cross-validate over bag chromosomes
    for val_chrom in bag_chroms:
        trial_setting = (
            f"{cell}_BOCO"
            f"_trial{trial.number}"
            f"_val{val_chrom}"
            f"_attn{attn}"
            f"_gw{gat_window}_ga{gat_alpha}"
            f"_dm{d_model}_h{n_heads}_el{e_layers}_dl{d_layers}"
            f"_ff{d_ff}_do{dropout:.3f}"
        )

        args = Namespace(
            # Data / CV config
            cell=cell,
            train_chroms=train_chroms_all,
            val_chroms=[val_chrom],
            test_chroms=[val_chrom],
            setting=trial_setting,
            checkpoints=os.path.join("checkpoints", trial_setting),

            # Tunables
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

            # GAT params (tunable)
            attn=attn,
            gat_window=gat_window,
            gat_alpha=gat_alpha,

            # Fixed
            batch_size=256,
            label_len=label_len,
            pred_len=1,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            activation='gelu',
            train_epochs=5,
            patience=3,
            lradj='type1',
            num_workers=8,
            use_multi_gpu=False,
            gpu=0,
            devices='0',
            selected_cols=[
                'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'
            ]
        )

        try:
            result = train_intra_cell_main(args)
            if isinstance(result, dict) and 'val_score' in result:
                scores.append(result['val_score'])
            else:
                scores.append(float("inf"))
        except Exception:
            # Penalize failed folds instead of crashing the whole study
            scores.append(float("inf"))
        finally:
            # Cleanup
            del args
            if 'result' in locals():
                del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 3) Aggregate across folds
    if not scores:
        return float("inf")

    return sum(scores) / len(scores)


def main():
    args = parse_args()
    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: optuna_objective_boco(trial, cell=args.cell, group_size=args.group_size),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    print("\n[✓] Best Hyperparameters Found (BOCO + GAT):")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    study_df = study.trials_dataframe()
    csv_path = f"optuna_BOCO_{args.cell}_GAT.csv"
    study_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved Optuna BOCO trials to: {csv_path}")


if __name__ == "__main__":
    main()