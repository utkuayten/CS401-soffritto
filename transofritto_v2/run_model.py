import sys, os
import json, argparse
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from transofritto.informer.exp.exp_informer import Exp_Informer

def run_model_main(args=None):
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
