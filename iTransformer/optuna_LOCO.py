import argparse
import optuna
import os
from argparse import Namespace
from train_intra_cell import main as train_intra_cell_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning with nested LOCO-CV for iTransformer"
    )
    parser.add_argument(
        '--cell', type=str, required=True,
        help="Cell name (e.g., mESC or H1)"
    )
    parser.add_argument(
        '--test_chrom', type=int, required=True,
        help="Chromosome held out for testing"
    )
    parser.add_argument('--val_chrom', type=int, required=True, help="Chromosome held out for validation")
    parser.add_argument(
        '--n_trials', type=int, default=20,
        help="Number of Optuna trials to run"
    )
    return parser.parse_args()


def get_all_chroms(cell: str):
    # Determine chromosome indices based on cell type
    if cell.lower().startswith('m'):
        return list(range(1, 20))
    else:
        return list(range(1, 23))


def optuna_objective(trial: optuna.Trial, cell: str, test_chrom: int) -> float:
    chroms_all = get_all_chroms(cell)
    inner_chroms = [c for c in chroms_all if c != test_chrom]
    scores = []

    for val_chrom in inner_chroms:
        train_chroms = [c for c in inner_chroms if c != val_chrom]
        trial_setting = f"{cell}_optunaVal{val_chrom}_test{test_chrom}_trial{trial.number}"

        # Build argument namespace for this fold
        args = Namespace(
            # Core settings
            cell=cell,
            is_training=1,
            model_id=trial_setting,
            model='iTransformer',
            data='custom',
            root_path='./iTransformer/data',
            data_path=f'{cell}_genomic.csv',
            features='M',
            target='target_1',
            freq='h',
            setting= trial_setting,
            checkpoints=os.path.join('checkpoints', trial_setting),
            exp_name='MTSF',

            # Chromosome splits
            train_chroms=train_chroms,
            val_chroms=[val_chrom],

            # Fixed data parameters

            enc_in=9,
            dec_in = 16,
            c_out=16,

            # Tunable model hyperparameters
            e_layers=trial.suggest_int('e_layers', 1, 3),
            d_layers=trial.suggest_int('d_layers', 1, 3),
            d_model=trial.suggest_categorical('d_model', [128, 256, 512]),
            n_heads=trial.suggest_categorical('n_heads', [2, 4, 8]),
            d_ff=trial.suggest_categorical('d_ff', [512, 1024, 2048]),
            dropout=trial.suggest_float('dropout', 0.05, 0.3),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
            seq_len=trial.suggest_categorical('seq_len',  [64, 128, 256]),
            label_len=seq_len//2,
            batch_size=64,

            # Fixed training parameters
            train_epochs=5,
            patience=3,
            lradj='type1',

            pred_len=1,

            # Model-specific defaults
            factor=1,
            embed='timeF',
            distil=False,
            des='test',
            class_strategy='projection',
            itr=1,
            loss='KL',
            output_attention=False,
            use_amp=False,
            use_gpu=True,
            gpu=0,
            use_multi_gpu=False,
            devices='0',
            num_workers=0,
            use_norm=0,
            inverse=False,
            moving_avg=25,
            do_predict=False,
            channel_independence=False,
            efficient_training=False,
            partial_start_index=0,
            activation='gelu'
        )

        # Execute training/validation for this fold
        result = train_intra_cell_main(args)
        if result and 'val_score' in result:
            scores.append(result['val_score'])
        else:
            print(f"[WARNING] No valid result returned for val_chrom {val_chrom}")

    # Return average validation score (minimize)
    return sum(scores) / len(scores) if scores else float('inf')


def main():
    args = parse_args()
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: optuna_objective(trial, args.cell, args.test_chrom),
        n_trials=args.n_trials
    )

    # Output best hyperparameters
    print("\n[✓] Best Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save all trials to CSV
    import pandas as pd
    study_df = study.trials_dataframe()
    csv_path = f"optuna_{args.cell}_iTransformer_test{args.test_chrom}.csv"
    study_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved Optuna trials to: {csv_path}")


if __name__ == '__main__':
    main()