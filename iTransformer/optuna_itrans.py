#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import optuna
from argparse import Namespace
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast


def objective(trial):
    # Unique identifier for this trial
    trial_setting = f"iTransformer_trial_{trial.number}"

    # Sample hyperparameters
    seq_len   = trial.suggest_categorical('seq_len', [64, 128, 256])
    label_len = seq_len // 2

    # Build the args namespace, matching your parser defaults
    args = Namespace(
        # Core settings
        cell='H1',
        is_training=1,
        model_id=trial_setting,
        model='iTransformer',
        data='custom',
        root_path='iTransformer/data/',
        data_path='H1_genomic.csv',
        features='M',
        target='target_1',
        freq='w',
        setting=trial_setting,
        checkpoints=os.path.join('../checkpoints', trial_setting),
        exp_name='MTSF',

        # Chromosome splits
        train_chroms=[1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22],
        val_chroms=[6],

        # Fixed data parameters
        enc_in=9,
        dec_in=16,
        c_out=16,

        # Tunable model hyperparameters
        e_layers=trial.suggest_int('e_layers', 1, 5),
        d_layers=trial.suggest_int('d_layers', 1, 5),
        d_model=trial.suggest_categorical('d_model', [128, 256, 512]),
        n_heads=trial.suggest_categorical('n_heads', [2, 4, 8]),
        d_ff=trial.suggest_categorical('d_ff', [512, 1024, 2048]),
        dropout=trial.suggest_float('dropout', 0.05, 0.3),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        use_norm=trial.suggest_categorical('use_norm', [True, False]),
        seq_len=seq_len,
        label_len=label_len,
        batch_size=64,

        # Fixed training parameters
        train_epochs=8,
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
        use_gpu=torch.backends.mps.is_available(),
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        num_workers=10,
        inverse=False,
        moving_avg=25,
        do_predict=False,
        channel_independence=False,
        efficient_training=False,
        partial_start_index=0,
        activation='gelu'
    )

    # Reproducibility
    args.device = torch.device('mps' if args.use_gpu else 'cpu')

    print(args)

    # Instantiate experiment and train
    exp = Exp_Long_Term_Forecast(args)
    exp.device = args.device
    exp.model.to(args.device)
    model, val_score = exp.train(args.setting)

    # Extract validation score (must match your train() return)
    if val_score is None:
        raise RuntimeError('Expected `val_score` in training metrics')

    return val_score


if __name__ == '__main__':
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    # Summarize best trial
    best = study.best_trial
    print(f"Best trial: #{best.number}")
    print(f"  Params: {best.params}")
    print(f"  Value:  {best.value}")

    # Save best parameters to JSON
    with open('best_params.json', 'w') as f:
        json.dump(best.params, f, indent=2)
