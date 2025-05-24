import sys, os
import json, argparse
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from transofritto.informer.exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='informer')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--setting', type=str, help ="setting for model")
parser.add_argument('--results_path', type=str, help='Directory to save results')

parser.add_argument('--root_path', type=str, help='root path for data')
parser.add_argument('--data_path', type=str, help='data path for input output combined')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=str, default='w')
parser.add_argument('--target', type=str, default='target_1')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint path')

parser.add_argument('--enc_in', type=int, default=9, help = "encoder input dimension")
parser.add_argument('--dec_in', type=int, default=16, help = "decoder input dimension")
parser.add_argument('--c_out', type=int, default=16)

parser.add_argument('--seq_len', type=int, help = "Give a sequence length.")
parser.add_argument('--label_len', type=int, help = "Give a label length.")
parser.add_argument('--pred_len', type=int, default=1, help = "prediction step")

parser.add_argument('--train_chroms', nargs='+', type=int, help='List of chromosomes for training')
parser.add_argument('--val_chroms', nargs='+', type=int,  help='List of chromosomes for validation')

parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
parser.add_argument('--e_layers', type=int, default=1)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.03)
parser.add_argument('--attn', type=str, default='prob')
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')


parser.add_argument('--learning_rate', type=float, default=0.000045)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--lradj', type=str, default='type1')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--use_multi_gpu', type=bool, default=False)
parser.add_argument('--use_amp', type=bool, default=False)
parser.add_argument('--output_attention', type=bool, default=False)
parser.add_argument('--inverse', type=bool, default=False)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--distil', type=bool, default=True)
parser.add_argument('--mix', type=bool, default=False)


def run_model_main(args=None):
    if args is None:

        args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        args.use_amp = True
        args.use_gpu = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        args.use_amp = False
        args.use_gpu = False
    else:
        device = torch.device('cpu')
        args.use_amp = False
        args.use_gpu = False
    args.device = device

    torch.set_printoptions(profile="full")
    torch.autograd.set_detect_anomaly(True)

    exp = Exp_Informer(args)
    model, val_score = exp.train(args.setting)
    exp.test(args.setting)

    params_path = os.path.join(args.checkpoints, args.setting)
    os.makedirs(params_path, exist_ok=True)

    args.device = str(args.device)  # Make serializable
    with open(os.path.join(params_path, f"{args.setting}_hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"[INFO] Hyperparameters saved to {params_path}/{args.setting}_hyperparameters.json")
    print(f"[INFO] Validation Score: {val_score}")
    return {"val_score": val_score}


if __name__ == '__main__':
    run_model_main()
