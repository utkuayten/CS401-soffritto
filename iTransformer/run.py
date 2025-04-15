import os
import sys

# Assuming run.py is inside the iTransformer directory,
# add the parent directory of iTransformer to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import torch
# Alternatively, if your iTransformer experiments are in a different module,
# adjust the import below accordingly:
# from transformer.informer.exp.exp_itransformer import Exp_iTransformer
import numpy as np

if __name__ == '__main__':
    from iTransformer.experiments.exp_itrans import Exp_iTransformer  # make sure this is implemented

    parser = argparse.ArgumentParser(description='iTransformer Experiment')

    # Data and model setup
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
    parser.add_argument('--data', type=str, default='custom',
                        help='dataset type')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options: [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding (ex: h for hourly)')
    parser.add_argument('--root_path', type=str, default='./data/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv',
                        help='data csv file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='directory for saving checkpoints')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
# Input/output dimensions: adjust these to your data
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=16, help='output size')

    parser.add_argument('--input_dim', type=int, default=10, help='input dimension (number of features) for the model')
    parser.add_argument('--output_dim', type=int, default=16, help='output dimension (number of target features) for the model')
    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # Architecture hyperparameters
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='number of decoder layers')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of ffn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--attn', type=str, default='prob', help='attention type')
    parser.add_argument('--factor', type=int, default=5, help='attention factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='type of time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')

    # Flags and miscellaneous
    parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')
    parser.add_argument('--output_attention', type=bool, default=False, help='whether to output attention weights')
    parser.add_argument('--inverse', type=bool, default=False, help='inverse output data')
    parser.add_argument('--padding', type=int, default=0,
                        help='padding type for decoder input: 0 for zeros, 1 for ones')
    parser.add_argument('--distil', type=bool, default=True, help='whether to use distillation')
    parser.add_argument('--mix', type=bool, default=True, help='whether to use mixing in output')

    # Column names used in the dataset, for example:
    parser.add_argument('--cols', type=list, default=[f'target_{i+1}' for i in range(16)],
                        help='list of target column names')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Parse arguments
    args = parser.parse_args()

    # MPS device support: if available, use Apple MPS; otherwise, fallback to CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    args.device = device

    # Define a setting string that uniquely identifies the experiment instance.
    setting = 'genomic_multitarget_itransformer'

    # Instantiate and run the experiment.
    exp = Exp_iTransformer(args)
    model = exp.train(setting)
    exp.test(setting)