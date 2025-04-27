import optuna
import torch
import argparse
import os, sys
import shutil
import gc

from optuna.pruners import ThresholdPruner
from optuna.exceptions import TrialPruned

# Adjust these according to your project
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer

def objective(trial):
    # Suggest hyperparameters
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
    embed = trial.suggest_categorical('embed', ["learned", "timeF"])

    # Set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=d_model)
    parser.add_argument('--e_layers', type=int, default=e_layers)
    parser.add_argument('--d_layers', type=int, default=d_layers)
    parser.add_argument('--n_heads', type=int, default=n_heads)
    parser.add_argument('--d_ff', type=int, default=d_ff)

    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--factor', type=int, default=factor)
    parser.add_argument('--embed', type=str, default=embed)
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--mix', type=bool, default=mix)
    parser.add_argument('--enc_in', type=int, default=9)
    parser.add_argument('--dec_in', type=int, default=16)
    parser.add_argument('--c_out', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=seq_len)
    parser.add_argument('--label_len', type=int, default=lab_len)
    parser.add_argument('--pred_len', type=int, default=1)

    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='target_1')
    parser.add_argument('--freq', type=str, default='w')
    parser.add_argument('--root_path', type=str, default='data')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--inverse', type=bool, default=False)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default='type1')

    parser.add_argument('--cols', type=list, default=[f'target_{i+1}' for i in range(16)])
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--train_epochs', type=int, default=7)

    # Parse args
    args = parser.parse_args(args=[])
    print('args', args)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device

    # Build experiment
    exp = Exp_Informer(args)

    # Train and get validation loss
    model, validation_loss = exp.train('genomic_multitarget_informer')

    # Report to Optuna
    trial.report(validation_loss, step=1)

    # Prune bad trials
    if trial.should_prune():
        print("Trial pruned by ThresholdPruner!")
        raise TrialPruned()

    gc.collect()
    return validation_loss

if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        pruner=ThresholdPruner(lower=None, upper=0.5)
    )

    study.optimize(objective, n_trials=7)

    # Print and save results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"Validation Loss: {trial.value}")
    print("Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df = study.trials_dataframe()
    df.to_csv("optuna_trials2.csv", index=False)

    best_n = study.best_trial.number
    src = f"trial_models/model_trial_{best_n}.pth"
    dst = "best_model.pth"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"\nBest model from trial #{best_n} saved to {dst}")
    else:
        print(f"\nModel checkpoint for best trial not found! (expected at {src})")
