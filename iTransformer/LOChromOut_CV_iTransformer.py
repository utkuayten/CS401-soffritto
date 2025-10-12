import argparse
from argparse import Namespace


from train_intra_cell import main as train_intra_cell_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested LOCO-CV for iTransformer: Outer test chrom + inner cross-validation on remaining chroms"
    )
    parser.add_argument('--cell', type=str, required=True,
                        help='Cell name (used to list available chroms)')
    parser.add_argument('--test_chrom', type=int, required=True,
                        help='Chromosome index to hold out as outer test')

    # iTransformer hyperparameters (must match those in run.py)
    parser.add_argument('--is_training', type=int, default=1,
                        help='1 = train mode, 0 = test mode')
    parser.add_argument('--model_id', type=str, default='exp1',
                        help='Experiment ID, used for naming')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='Model type (e.g., iTransformer)')
    parser.add_argument('--data', type=str, default='custom',
                        help='Dataset identifier')
    parser.add_argument('--root_path', type=str, default='./iTransformer/data',
                        help='Root path of the data files')
    parser.add_argument('--data_path', type=str, default='data.csv',
                        help='CSV file with the time series data')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task: M, S, or MS')
    parser.add_argument('--target', type=str, default='target_1',
                        help='Target feature in S/MS tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency of the time series')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Directory for model checkpoints')
    parser.add_argument('--exp_name', type=str, default='MTSF',
                        help='Experiment name')

    # Chromosome folds will override these two dynamically
    parser.add_argument('--train_chroms', nargs='+', type=int,
                        help='List of chroms for training (overridden by CV script)')
    parser.add_argument('--val_chroms', nargs='+', type=int,
                        help='List of chroms for validation (overridden by CV script)')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Start token length for decoder')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='Prediction sequence length')

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=9,
                        help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=16, help='output size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Dimension of FFN layer')
    parser.add_argument('--factor', type=int, default=5,
                        help='Factor for ProbSparse attention')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Embedding type')
    parser.add_argument('--distil', action='store_true', default=False,
                        help='Use distilling in transformer')
    parser.add_argument('--des', type=str, default='test',
                        help='Experiment description')
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='Strategy for classification token')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--use_norm', type=int, default=0, help='use norm and denorm')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # --- Wavelet options ---
    parser.add_argument('--use_wavelet', action='store_true', help='Enable wavelet features (e.g., SWT) on inputs')
    parser.add_argument('--wavelet_name', type=str, default='db4', help='PyWavelets wavelet name (e.g., db4, coif1, sym4)')
    parser.add_argument('--wavelet_levels', type=int, default=1, help='Number of decomposition levels (>=1)')
    parser.add_argument('--keep_original', action='store_true', help='Concatenate original features with wavelet bands')
    parser.add_argument('--wavelet_where', type=str, default='dataset', choices=['dataset','model'], help='Where to apply wavelet transform')


    # Training hyperparameters
    parser.add_argument('--itr', type=int, default=1,
                        help='Number of training loops')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Learning rate scheduler')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

    # GPU settings
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Primary GPU id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0', help='Comma-separated GPU device ids')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    return parser.parse_args()


def get_available_chroms(cell):
    # Mirror behavior from LOChromOut_CV_informer
    return list(range(1, 20)) if cell.startswith('m') else list(range(1, 22))


def main():
    args = parse_args()
    outer_test_chrom = args.test_chrom
    chroms_all = get_available_chroms(args.cell)

    if outer_test_chrom not in chroms_all:
        raise ValueError(f"[ERROR] test_chrom {outer_test_chrom} is not in {chroms_all}")

    inner_chroms = [c for c in chroms_all if c != outer_test_chrom]
    print(f"\n[INFO] Nested LOCO: Outer test chrom = {outer_test_chrom}")
    print(f"[INFO] Inner LOCO folds on: {inner_chroms}")

    results = []
    for inner_val_chrom in inner_chroms:
        inner_train_chroms = [c for c in inner_chroms if c != inner_val_chrom]
        print(f"\n[INFO] Inner Fold: Train on {inner_train_chroms} | Validate on {inner_val_chrom}")
        # Build run arguments for this fold
        run_args = Namespace(
            cell=args.cell,
            is_training=args.is_training,
            model_id=f"{args.cell}_outer{outer_test_chrom}_val{inner_val_chrom}",
            model=args.model,
            data=args.data,
            root_path=args.root_path,
            data_path=f'{args.cell}_genomic.csv',
            features=args.features,
            target=args.target,
            freq=args.freq,
            checkpoints=args.checkpoints,
            exp_name=args.exp_name,
            train_chroms=inner_train_chroms,
            val_chroms=[inner_val_chrom],
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            factor=args.factor,
            embed=args.embed,
            distil=args.distil,
            des=args.des,
            class_strategy=args.class_strategy,
            itr=args.itr,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_epochs=args.train_epochs,
            loss=args.loss,
            lradj=args.lradj,
            use_amp=args.use_amp,
            use_gpu=args.use_gpu,
            gpu=args.gpu,
            use_multi_gpu=args.use_multi_gpu,
            devices=args.devices,
            num_workers=args.num_workers,
            output_attention=args.output_attention,
            use_norm=args.use_norm,
            dropout=args.dropout,
            activation=args.activation,
            patience=args.patience,
            inverse=args.inverse,
            moving_avg=args.moving_avg,
            do_predict=args.do_predict,
            channel_independence=args.channel_independence,
            efficient_training =args.efficient_training,
            partial_start_index=args.partial_start_index,
            use_wavelet=args.use_wavelet,
            wavelet_name=args.wavelet_name,
            wavelet_levels=args.wavelet_levels,
            keep_original=args.keep_original,
            wavelet_where=args.wavelet_where,
        )

        # Execute training/validation for this fold
        result = train_intra_cell_main(run_args)
        if result and 'val_score' in result:
            results.append((inner_val_chrom, result['val_score']))
            print(f"[INFO] Val Chrom {inner_val_chrom}: {result['val_score']:.4f}")
        else:
            print(f"[WARNING] No valid result returned for val_chrom {inner_val_chrom}")

    # Summary of inner folds
    print(f"\n[SUMMARY] Inner Validation Scores (Outer Test Chrom: {outer_test_chrom})")
    for chrom, score in results:
        print(f"  Val Chrom {chrom}: {score:.4f}")

    if results:
        avg_score = sum(score for _, score in results) / len(results)
        print(f"\n[AVERAGE INNER VAL SCORE]: {avg_score:.4f}")
    else:
        print("\n[ERROR] No results from any inner fold.")


if __name__ == "__main__":
    main()
