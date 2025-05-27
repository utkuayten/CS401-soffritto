import argparse
import optuna
from argparse import Namespace
from train_intra_cell import main as train_intra_cell_main
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning with nested LOCO-CV")
    parser.add_argument('--cell', type=str, required=True, help="Cell name (e.g. mESC or H1)")
    parser.add_argument('--test_chrom', type=int, required=True, help="Chromosome held out for testing")
    parser.add_argument('--n_trials', type=int, default=10, help="Number of Optuna trials to run")
    return parser.parse_args()

def get_all_chroms(cell):
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 23))

def optuna_objective(trial, cell, test_chrom):
    chroms_all = get_all_chroms(cell)
    inner_chroms = [c for c in chroms_all if c != test_chrom]

    scores = []
    for val_chrom in inner_chroms:
        train_chroms = [c for c in inner_chroms if c != val_chrom]
        trial_setting = f"{cell}_optunaVal{val_chrom}_test{test_chrom}_trial{trial.number}"

        args = Namespace(
            cell=cell,
            train_chroms=train_chroms,
            val_chroms=[val_chrom],
            setting=trial_setting,
            checkpoints=os.path.join("checkpoints", trial_setting),

            # Tunable
            e_layers=trial.suggest_int("e_layers", 1, 2),
            d_layers=trial.suggest_int("d_layers", 1, 2),
            d_model=trial.suggest_categorical("d_model", [128, 256, 512]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            d_ff=trial.suggest_categorical("d_ff", [512, 1024, 2048]),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

            # Fixed
            seq_len=32,
            label_len=16,
            pred_len=1,
            enc_in=9,
            dec_in=16,
            c_out=16,
            activation='gelu',
            attn='prob',
            factor=5,
            train_epochs=5,
            patience=3,
            lradj='type1',
            num_workers=4,
            use_multi_gpu=False,
            gpu=0,
            devices='0'
        )

        result = train_intra_cell_main(args)
        if result and 'val_score' in result:
            scores.append(result['val_score'])

    return sum(scores) / len(scores) if scores else 0.0

def main():
    args = parse_args()
    study = optuna.create_study(direction='minimize')  # KL divergence: minimize
    study.optimize(lambda trial: optuna_objective(trial, args.cell, args.test_chrom), n_trials=args.n_trials)

    print("\n[✓] Best Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save all trial results to CSV
    import pandas as pd
    study_df = study.trials_dataframe()
    csv_path = f"optuna_{args.cell}_test{args.test_chrom}.csv"
    study_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved Optuna trials to: {csv_path}")

if __name__ == "__main__":
    main()