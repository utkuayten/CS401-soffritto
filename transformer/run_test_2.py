if __name__ == '__main__':
    import argparse
    import torch
    from transformer.informer.exp.exp_informer import Exp_Informer

    parser = argparse.ArgumentParser()

    # Data and model setup
    parser.add_argument('--model', type=str, default='informer')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='target_1')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--root_path', type=str, default='../data/')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Input/output dimensions
    parser.add_argument('--enc_in', type=int, default=16)
    parser.add_argument('--dec_in', type=int, default=16)
    parser.add_argument('--c_out', type=int, default=16)

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=48)

    # Architecture
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)

    # Flags
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--inverse', type=bool, default=False)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--mix', type=bool, default=True)

    # Column names
    parser.add_argument('--cols', type=list, default=[
        *[f'target_{i+1}' for i in range(16)]
    ])

    # Parse args
    args = parser.parse_args(args=[])

    # MPS device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    args.device = device

    # Run
    setting = 'genomic_multitarget_informer'
    exp = Exp_Informer(args)
    model = exp.train(setting)
    exp.test(setting)
