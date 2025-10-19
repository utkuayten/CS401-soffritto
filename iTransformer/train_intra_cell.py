import argparse
import subprocess
import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from run import main as run_model_main

def parse_args():
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./iTransformer/data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='target_1', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # genomic arguments
    parser.add_argument('--train_chroms', nargs='+', type=int, help='List of chromosomes for training',
                        default={1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
    parser.add_argument('--val_chroms', nargs='+', type=int,  help='List of chromosomes for validation',
                        default=[9])
    #parser.add_argument('--test_chroms', nargs='+', type=int, help='List of chromosomes for testing',
    #                    default=[9])

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=64, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=16, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.129379, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', )

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.000185217, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='KL', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # --- Wavelet options ---
    parser.add_argument('--use_wavelet', action='store_true', help='Enable wavelet features (e.g., SWT) on inputs')
    parser.add_argument('--wavelet_name', type=str, default='db4', help='PyWavelets wavelet name (e.g., db4, coif1, sym4)')
    parser.add_argument('--wavelet_levels', type=int, default=1, help='Number of decomposition levels (>=1)')
    parser.add_argument('--keep_original', action='store_true', help='Concatenate original features with wavelet bands')
    parser.add_argument('--wavelet_where', type=str, default='dataset', choices=['dataset','model'], help='Where to apply wavelet transform')


    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=int, default=False, help='use norm and denorm')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--setting', type=str, default='best_params_run', help='setting')
    return parser.parse_args()

def main(args=None):

    if args is None:
        args = parse_args()

    # Command to run the script with arguments
    script_path = os.path.join(os.path.dirname(__file__), 'run.py')
    data_dir = os.path.join(os.path.dirname(__file__), 'data/raw')
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')


    if not args.setting:
        val_str = "-".join(str(c) for c in args.val_chroms)
        args.setting = f"{args.cell}_val_{val_str}"

    print(f'Encoder input,Decoder input {args.enc_in}')

    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    main()
