import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def build_parser():
    parser = argparse.ArgumentParser(
        description='PatchTST / Formers training runner (genomic variant with chrom splits)'
    )

    # -------------------- ORIGINAL ARGS (UNCHANGED / ALL KEPT) --------------------
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [Autoformer, Informer, Transformer, PatchTST]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv', help='data file (or will be set from --cell)')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='target_1', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=16, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # PatchTST (kept as in original)
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=2, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=10, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=1,
                        help='0: default 1: value+temporal+positional 2: value+temporal 3: value+positional 4: value only')
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=16, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.03, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # -------------------- NEW ARGS (ADDED FROM INFORMER GENOMIC RUNNER) --------------------
    parser.add_argument('--setting', type=str, default=None, help='explicit run setting name')
    parser.add_argument('--cell', type=str, required=False, help='cell name to derive data path (e.g., H1)')
    parser.add_argument('--train_chroms', nargs='+', type=int, help='List of chromosomes for training',
                        default={1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
    parser.add_argument('--val_chroms', nargs='+', type=int,  help='List of chromosomes for validation',
                        default=[6])
    parser.add_argument('--attn', type=str, default='prob', help='attention type (if a Former uses it)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='optimizer weight decay')

    # Wavelet options (safe no-ops if you don’t use them downstream)
    parser.add_argument('--use_wavelet', action='store_true',
                        help='Enable wavelet features (e.g., SWT) on inputs')
    parser.add_argument('--wavelet_name', type=str, default='db4',
                        help='PyWavelets wavelet name')
    parser.add_argument('--wavelet_levels', type=int, default=1,
                        help='Number of decomposition levels (>=1)')
    parser.add_argument('--keep_original', action='store_true',
                        help='Concatenate original features with wavelet bands')
    parser.add_argument('--wavelet_where', type=str, default='dataset',
                        choices=['dataset', 'model'],
                        help='Where to apply wavelet transform')

    # Feature selection
    parser.add_argument('--selected_cols', nargs='+', type=str,
                        default=['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                                 'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage','date'],
                        help='Columns to use as inputs')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # -------------------- Repro --------------------
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # -------------------- Device --------------------
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # -------------------- Derived paths & constants (genomic) --------------------
    # If a cell is provided, derive paths like the Informer runner; otherwise keep user-provided paths.
    base_dir = os.path.dirname(__file__)
    default_root = os.path.join(base_dir, "data")
    default_ckpt = os.path.join(base_dir, "checkpoints")
    default_results = os.path.join(base_dir, "results")

    # Normalize root/checkpoints
    if args.root_path in (None, './data/ETT/', './data'):
        args.root_path = default_root
    if args.checkpoints in (None, './checkpoints/'):
        args.checkpoints = default_ckpt
    args.results_path = default_results  # handy for saving metrics

    # If user provided --cell, prefer {root}/{cell}_genomic.csv
    if getattr(args, 'cell', None):
        args.data_path = os.path.join(args.root_path, f"{args.cell}_genomic.csv")
        # For convenience, enforce consistent defaults for our genomic tasks
        if args.data == 'custom':
            args.freq = "w"
            args.embed = "timeF"
            args.output_attention = False
            args.distil = False

    # -------------------- Setting string --------------------
    if not args.setting:
        if getattr(args, 'cell', None) and args.val_chroms is not None:
            val_str = "-".join(str(c) for c in args.val_chroms) if len(args.val_chroms) else "none"
            args.setting = f"{args.cell}_val_{val_str}"
        else:
            # fall back to the legacy rich setting string
            args.setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
                args.distil, args.des
            )

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # If the user already provided a setting, reuse it; otherwise append the iteration idx
            if args.setting:
                setting = f"{args.setting}_{ii}"
            else:
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
                    args.distil, args.des, ii
                )

            exp = Exp(args)  # set experiments
            print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.do_predict:
                print(f'>>>>>>> predicting : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = args.setting or '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii
        )
        exp = Exp(args)  # set experiments
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()