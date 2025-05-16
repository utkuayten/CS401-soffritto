# tune_itransformer.py

import os
import argparse
import torch
import optuna

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

class Objective:
    def __init__(self, base_args):
        self.base_args = base_args

    def __call__(self, trial):
        # 1) Clone base args
        args = argparse.Namespace(**vars(self.base_args))

        # 2) Fill in any args that Exp_Long_Term_Forecast/__init__ expects:
        args.output_attention = False   # <-- was missing
        args.use_amp         = False   # if you don’t use mixed‑precision
        args.patience        = 3       # EarlyStopping patience
        args.train_epochs    = 20      # total epochs per trial
        args.lradj           = 'type1' # keep default scheduler
        args.factor          = 3
        # you can also set weight decay, scheduler type, etc. here if needed

        # 3) Optuna search space
        args.e_layers       = trial.suggest_int(     'e_layers',    1,    8)
        args.d_layers       = trial.suggest_int(     'd_layers',    1,    8)
        args.d_model        = trial.suggest_categorical('d_model',   [64, 128, 256, 512])
        args.n_heads        = trial.suggest_categorical('n_heads',   [2, 4, 8])
        args.d_ff           = trial.suggest_categorical('d_ff',      [128, 256, 512, 1024, 2048])
        args.dropout        = trial.suggest_uniform(     'dropout',    0.0,  0.5)
        args.learning_rate  = trial.suggest_loguniform( 'learning_rate', 1e-5, 1e-3)
        args.batch_size     = trial.suggest_categorical('batch_size',[16, 32, 64])

        # 4) Fixed data/model settings
        args.device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        args.root_path   = '/Users/ozgun/DataspellProjects/CS401-soffritto/transofritto/data'
        args.data_path   = 'H1_genomic.csv'
        args.checkpoints = './checkpoints/optuna'
        args.model       = 'iTransformer'
        args.features    = 'M'
        args.target      = 'target_1'
        args.freq        = 'w'
        args.seq_len     = 32
        args.label_len   = 16
        args.pred_len    = 1
        args.padding     = 0
        args.distil      = True
        args.mix         = True
        args.use_gpu     = True
        args.use_norm    = False
        args.embed       = 'timeF'
        args.class_strategy = 'cls_token'
        args.activation  = 'gelu'
        args.use_multi_gpu = False
        args.data        = 'custom'
        args.num_workers = 4
        args.use_gpu = False
        args.gpu = 'mps'
        args.val_chroms = [1]
        args.train_chroms = [1]
        args.use_norm = False

        # 5) Run one trial
        exp = Exp_Long_Term_Forecast(args)
        exp.train('optuna_itransformer')

        # 6) Compute validation loss
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        val_loss = exp.vali(vali_data, vali_loader, criterion)

        return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='number of Optuna trials')
    cli_args = parser.parse_args()

    # build a minimal base_args just to carry flags around
    base_args = argparse.Namespace()

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler())
    study.optimize(Objective(base_args), n_trials=cli_args.n_trials)

    print("Best validation loss:", study.best_value)
    print("Best hyperparameters:", study.best_params)


    # Convert the trials to a DataFrame
    df = study.trials_dataframe()

    # Save to CSV (or any other format you like)
    df.to_csv("optuna_trials.csv", index=False)