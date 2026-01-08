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

        seq_len_value = trial.suggest_categorical("seq_len", [32, 64])
        label_len_value = seq_len_value // 2

        args = Namespace(
            cell=cell,
            train_chroms=train_chroms,
            val_chroms=[val_chrom],
            test_chroms =[val_chrom],
            setting=trial_setting,
            checkpoints=os.path.join("checkpoints", trial_setting),

            # Smaller to avoid OOM
            e_layers=trial.suggest_int("e_layers", 1, 3),
            d_layers=trial.suggest_int("d_layers", 1, 3),
            d_model=trial.suggest_categorical("d_model", [128, 256, 512]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
            d_ff=trial.suggest_categorical("d_ff", [256, 512, 1024]),
            dropout=trial.suggest_float("dropout", 0.05, 0.15),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            factor=trial.suggest_categorical("factor", [3, 5]),

            seq_len=seq_len_value,
            label_len=label_len_value,
            pred_len=1,

            batch_size=1024,   # OOM safe
            enc_in=9,
            dec_in=16,
            c_out=16,
            activation='gelu',
            attn='prob',
            train_epochs=5,
            patience=3,
            lradj='type1',
            num_workers=2,
            use_multi_gpu=False,
            gpu=0,
            devices='0',
            selected_cols=['H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me3','H3K9me3','GC_content','gene_density','2-stage']
        )

        # ---- TRAIN ----
        result = train_intra_cell_main(args)

        if result and 'val_score' in result:
            scores.append(result['val_score'])

        # ---- GPU CLEANUP ----
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    return sum(scores) / len(scores) if scores else 0.0

def main():
    args = parse_args()
    study = optuna.create_study(direction='minimize')  # KL divergence: minimize
    study.optimize(lambda trial: optuna_objective(trial, args.cell, args.test_chrom)
                   , n_trials=args.n_trials, n_jobs = 8)

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