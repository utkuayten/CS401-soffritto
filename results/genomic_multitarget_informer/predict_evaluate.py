import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys, os
#from transformer.informer.utils.metrics import metric
# Add the project root (2 directories up) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class InferenceModel:
    def __init__(self, checkpoint_path):
        """
        :param checkpoint_path: Path to the trained checkpoint (e.g. 'checkpoints/checkpoint.pth').
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        """
        Load your trained Informer model from checkpoint.
        Adjust the model instantiation to your actual model architecture.
        """
        import argparse

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
        # Fix for interactive environments
        args = parser.parse_args(args=[])

        from transformer.informer.exp.exp_informer import Exp_Informer
        setting = 'genomic_multitarget_informer'

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(device)
        print("MPS available:", torch.backends.mps.is_available())

        args.device = device
        args.use_amp = False
        torch.set_printoptions(profile="full")
        torch.autograd.set_detect_anomaly(True)

        model = Exp_Informer(args)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check if checkpoint is a dictionary with the key 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        return model

    def predict(self, data_loader):
        """
        :param data_loader: A PyTorch DataLoader yielding batches.
        :return: Numpy array of all predictions concatenated.
        """
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # Adjust this line based on your batch structure
                batch_x = batch[0].to(self.device)
                preds = self.model(batch_x)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    def evaluate(self, y_true, y_pred):
        """
        Compute and print metrics: MAPE, MSE, RMSE, MAE, and R2.
        :param y_true: Ground truth array (Numpy).
        :param y_pred: Predicted array (Numpy).
        """
        results = self.compute_metrics(y_true, y_pred)
        return results


    def compute_metrics(self, y_true, y_pred):
        """
        Computes metrics given 3D arrays (samples, timesteps, targets).
        We use a threshold for MAPE computation to avoid huge errors when y_true is near zero.
        """
        # Reshape to (samples * timesteps, targets)
        y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
        y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

        # Standard epsilon to avoid division by zero
        epsilon = 1e-9

        # If many y_true values are close to zero, using them directly can blow up MAPE.
        # Here we use a threshold: if |y_true| < threshold, use threshold instead.
        threshold = 1e-3  # Adjust this based on the scale of your data
        denominator = np.where(np.abs(y_true_reshaped) < threshold, threshold, np.abs(y_true_reshaped))
        mape_per_target = np.mean(np.abs((y_true_reshaped - y_pred_reshaped) / denominator), axis=0) * 100
        mape_overall = np.mean(mape_per_target)

        mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
        r2 = r2_score(y_true_reshaped, y_pred_reshaped)

        print("MAPE per target:", mape_per_target)
        print("Overall MAPE: {:.4f}".format(mape_overall))
        print("MSE: {:.4f}".format(mse))
        print("RMSE: {:.4f}".format(rmse))
        print("MAE: {:.4f}".format(mae))
        print("R2: {:.4f}".format(r2))

        return {
            "MAPE_per_target": mape_per_target,
            "MAPE_overall": mape_overall,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }


    def evaluate_from_files(self, true_file_path, pred_file_path):
        """
        Load ground truth and prediction arrays from .npy files and evaluate.
        """
        y_true = np.load(true_file_path)
        y_pred = np.load(pred_file_path)


        return self.evaluate(y_true, y_pred)