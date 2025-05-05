import os
import sys
import shutil
import gc
import argparse
import torch
import optuna
from argparse import Namespace
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer

def objective(trial):
    setting = f"trial_{trial.number}"

    # Hyperparameter search
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.15, 0.30)
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    e_layers = trial.suggest_int('e_layers', 1, 2)
    d_layers = trial.suggest_int('d_layers', 1, 2)
    seq_len = trial.suggest_int('seq_len', 20, 50)
    lab_len = int(seq_len * 0.5)
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    d_ff = trial.suggest_categorical('d_ff', [512, 1024])
    factor = trial.suggest_categorical('factor', [3, 5, 7])
    mix = trial.suggest_categorical('mix', [True, False])
    embed = "timeF"

    data_dir = os.path.join(THIS_DIR, 'data')
    checkpoints_dir = os.path.join(THIS_DIR, 'checkpoints')
    cell = "H1"
    train_chroms = [1, 2, 3, 4, 5, 6]
    val_chroms = [7]

    args = Namespace(
        model='informer',
        setting='multitarget',
        data='custom',
        root_path=data_dir,
        data_path=f"{cell}_genomic.csv",
        target='target_1',
        features='M',
        freq='w',
        enc_in=9,
        dec_in=16,
        c_out=16,
        seq_len=seq_len,
        label_len=lab_len,
        pred_len=1,
        train_epochs=10,
        batch_size=128,
        learning_rate=learning_rate,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        checkpoints=checkpoints_dir,
        weight_decay=0.001,
        dropout=dropout,
        d_model=d_model,
        e_layers=e_layers,
        d_layers=d_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        attn='prob',
        factor=factor,
        embed=embed,
        activation='gelu',
        mix=mix,
        num_workers=4,
        gpu=0,
        device=0,
        output_attention=False,
        inverse=False,
        use_multi_gpu=False,
        padding=0,
        distil=True,
        lradj='type1',
        patience=3,
    )

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.use_amp = True
        args.use_gpu = True
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
        args.use_amp = False
        args.use_gpu = False
    else:
        args.device = torch.device('cpu')
        args.use_amp = False
        args.use_gpu = False

    exp = Exp_Informer(args)
    model, validation_losses = exp.train(setting)
    validation_loss = min(validation_losses)

    trial.report(validation_loss, step=1)
    if trial.should_prune():
        raise TrialPruned()

    os.makedirs('trial_models', exist_ok=True)
    ckpt_src = os.path.join(args.checkpoints, setting, f'checkpoint_{args.model}.pth')
    ckpt_dst = os.path.join('trial_models', f'model_trial_{trial.number}.pth')
    if os.path.exists(ckpt_src):
        shutil.copy(ckpt_src, ckpt_dst)
    else:
        print(f"[WARN] missing checkpoint: {ckpt_src}")

    gc.collect()
    return validation_loss

if __name__ == '__main__':
    storage = 'sqlite:///optuna_genomic.db'
    study = optuna.create_study(
        storage=storage,
        study_name='genomic_multitarget',
        load_if_exists=True,
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(),
    )

    study.optimize(objective, n_trials=30, n_jobs=10)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Loss: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    df = study.trials_dataframe()
    df.to_csv('optuna_trials2.csv', index=False)

    os.makedirs('trial_models', exist_ok=True)
    best_ckpt = os.path.join('trial_models', f'model_trial_{best.number}.pth')
    if os.path.exists(best_ckpt):
        shutil.copy(best_ckpt, 'best_model.pth')
        print('Best model saved to best_model.pth')
    else:
        print(f"[WARN] best checkpoint not found: {best_ckpt}")
