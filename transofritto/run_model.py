import argparse
import sys
import os

if __name__ == '__main__':
    from informer.exp.exp_informer import Exp_Informer
    import torch

    parser = argparse.ArgumentParser()
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


    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--label_len', type=int, default=2)
    parser.add_argument('--pred_len', type=int, default=1)

    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512*4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--factor', type=int, default=5)      # ← add this line
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--train_epochs', type=int, default=2)
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
    parser.add_argument('--mix', type=bool, default=True)

    parser.add_argument('--cols', type=list, default=[
        *[f'target_{i+1}' for i in range(16)]
    ])

    # Fix for interactive environments
    args = parser.parse_args(args=[])


    setting = 'genomic_multitarget_informer'

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.device = device
    args.use_amp = False
    torch.set_printoptions(profile="full")
    torch.autograd.set_detect_anomaly(True)

    exp = Exp_Informer(args)
    model,val_score = exp.train(setting)
    exp.test(setting)
    print(val_score)