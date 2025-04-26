import os, sys
# assume this file lives two levels under your project root,
# adjust the number of '..' if needed
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)
import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np

# allow imports from this folder
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from predict_evaluate import InferenceModel

# no need to re-import Dataset; PredictionModel handles it

def main():
    parser = argparse.ArgumentParser(description="Run Informer checkpoint inference and evaluation")
    parser.add_argument(
        '--checkpoint', type=str,
        default='checkpoints/genomic_multitarget_informer/checkpoint.pth',
        help='Path to saved .pth checkpoint'
    )
    parser.add_argument(
        '--setting', type=str,
        default='genomic_multitarget_informer',
        help='Name of experiment setting (used if you fallback to exp.test)'
    )
    # Optional override of loader parameters if desired
    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='target_1')
    parser.add_argument('--freq', type=str, default='w')
    parser.add_argument('--root_path', type=str, default='data')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--enc_in', type=int, default=9)   # number of input features
    parser.add_argument('--dec_in', type=int, default=16)   # decoder input feature dim (target count)
    parser.add_argument('--c_out', type=int, default=16)    # number of output targets to predict


    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.14)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--factor', type=int, default=7)      # ← add this line
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--learning_rate', type=float, default=0.000045)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1')

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
    parser.add_argument('--mix', type=bool, default=False)

    parser.add_argument('--cols', type=list, default=[
        *[f'target_{i+1}' for i in range(16)]
    ])
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device
    args.use_amp = False
    # Instantiate and load model
    inf = InferenceModel(args, checkpoint_path=args.checkpoint)

    # 1) Option A: If you already have true/pred .npy files saved by exp.test(), use:
    # metrics = inf.evaluate_from_files(
    #     true_file_path=f"results/{args.setting}/true.npy",
    #     pred_file_path=f"results/{args.setting}/pred.npy"
    # )
    # print(metrics)
    # return

    # 2) Option B: Re-run inference & eval
    preds, trues = inf.predict(flag='test')
    pred_probs = np.exp(preds).squeeze(1)
    pred_probs /= pred_probs.sum(axis=1, keepdims=True)
    true_probs = trues.squeeze(1)
    metrics = inf.evaluate(true_probs, pred_probs)
    print("Evaluation metrics on test set:", metrics)

if __name__ == '__main__':
    main()
