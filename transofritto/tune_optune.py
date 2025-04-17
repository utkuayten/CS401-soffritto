import optuna
import torch
import argparse
from informer.exp.exp_informer import Exp_Informer

def objective(trial):
    # Suggest hyperparameters using the trial object.
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.01, 0.15)
    d_model = trial.suggest_categorical('d_model', [128 ,256, 512])
    e_layers = trial.suggest_int('e_layers', 1, 4)
    d_layers = trial.suggest_int('d_layers', 2, 6)
    n_heads = trial.suggest_categorical('n_heads', [2,4,8])
    d_ff = trial.suggest_categorical('d_ff', [512,1024,2048])
    factor = trial.suggest_categorical('factor', [3,5,7])
    mix = trial.suggest_categorical('mix', [True, False])

    # Create an argparse.ArgumentParser and set values from the trial suggestions.
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
    parser.add_argument('--embed', type=str, default='fixed')
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--mix', type=bool, default=mix)
    parser.add_argument('--enc_in', type=int, default=9)   # number of input features
    parser.add_argument('--dec_in', type=int, default=16)   # decoder input feature dim (target count)
    parser.add_argument('--c_out', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--label_len', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=1)

    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='target_1')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--root_path', type=str, default='/users/ozgun/DataspellProjects/CS401-soffritto/data')
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

    parser.add_argument('--cols', type=list, default=[
        *[f'target_{i+1}' for i in range(16)]
    ])

    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--train_epochs', type=int, default=10)

    # For interactive environments, override command-line arguments.
    args = parser.parse_args(args=[])
    print('args',args)
    # Set your device (example: check if a GPU accelerator is available).
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device

    # Instantiate your experiment class which builds the model and handles training.
    exp = Exp_Informer(args)

    # Train for a small number of epochs to quickly evaluate hyperparameters.
    # This method should return the validation loss as a metric.
    model, validation_loss = exp.train('genomic_multitarget_informer')

    # Report intermediate values if you want to use pruning (optional):
    trial.report(validation_loss, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return validation_loss

if __name__ == '__main__':
    # Create a study object. Set the direction to 'minimize' if you are minimizing loss.
    study = optuna.create_study(direction='minimize')

    # Optimize the objective function. Adjust n_trials according to available resources.
    study.optimize(objective, n_trials=10)

    # Print the results of the best trial.
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Convert the trials to a DataFrame
    df = study.trials_dataframe()

    # Save to CSV (or any other format you like)
    df.to_csv("optuna_trials.csv", index=False)
