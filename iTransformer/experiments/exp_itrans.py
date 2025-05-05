import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your iTransformer model from the module where the iTransformer class is defined.
#from iTransformer.model.iTransformer import iTransformer
# Import your dataset classes from the data_provider (make sure these files are set up correctly for genomic data)
from iTransformer.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from iTransformer.utils.metrics import metric
from iTransformer.utils.tools import EarlyStopping, adjust_learning_rate
from iTransformer.experiments.exp_basic import Exp_Basic
from soffritto.train_leave_one_cell_line_out import test_data


class Exp_iTransformer(Exp_Basic):
    def __init__(self, args):
        # Your args should contain parameters like:
        #   input_dim, output_dim, seq_len, label_len, pred_len, factor, d_model, n_heads,
        #   e_layers, d_layers, d_ff, dropout, attn, embed, freq, activation,
        #   output_attention, distil, mix, use_amp, inverse, padding, use_norm, class_strategy, etc.
        super(Exp_iTransformer, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        # Choose a dataset based on the data argument; for genomic data, you likely use your custom dataset.
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        # For prediction mode:
        if flag == 'pred':
            Data = Dataset_Pred
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq if hasattr(args, 'detail_freq') else args.freq
        else:
            Data = data_dict.get(args.data, Dataset_Custom)
            shuffle_flag = True if flag == 'train' else False
            drop_last = True if flag in ['train', 'test'] else False
            batch_size = args.batch_size
            freq = args.freq

        # For genomic data you may have specific settings (like input features, target columns, etc.)
        # Here we assume that the Data class accepts these parameters.
        dataset = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=0,    # Since in our genomic pipeline we removed extra timeenc if needed
            freq=freq,
            cols=args.cols
        )
        print(f"{flag} data length: {len(dataset)}")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return dataset, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        # Using KLDivLoss (which expects log-probabilities as input and probability distributions as targets)
        return nn.KLDivLoss(reduction='batchmean', log_target=False)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in vali_loader:
                # Transfer data to device
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Create decoder input: use the first label_len of batch_y and pad with zeros or ones for pred_len
                if self.args.padding == 0:
                    pad = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                else:
                    pad = torch.ones((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], pad], dim=1)
                # Forward pass (assumes model output applies log_softmax)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # If your model returns attention as a tuple, take the first element
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                # Optionally apply inverse transform if targets are normalized
                if vali_data.scale and self.args.inverse:
                    outputs = vali_data.inverse_transform(outputs)
                # Select target: we use the last pred_len time steps and the relevant features
                f_dim = -1 if self.args.features == 'MS' else 0
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, target)
                losses.append(loss.item())
        self.model.train()
        return np.average(losses)

    def train(self, setting):
        # Load data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model = self._build_model()
        self.model.to(self.device)
        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        time_now = time.time()
        total_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        all_val_losses = []

        for epoch in range(self.args.train_epochs):
            epoch_losses = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                # Transfer data to device
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # Create decoder input
                if self.args.padding == 0:
                    pad = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                else:
                    pad = torch.ones((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], pad], dim=1)

                # Forward pass with optional automatic mixed precision
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                        # If inverse transformation is desired during training:
                        if self.args.inverse:
                            outputs = train_data.inverse_transform(outputs)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        target = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    if self.args.inverse:
                        outputs = train_data.inverse_transform(outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    target = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

                epoch_losses.append(loss.item())

                # Optionally log every fixed number of iterations
                if (i+1) % 100 == 0:
                    iter_speed = (time.time() - time_now) / (i+1)
                    print(f"Epoch {epoch+1}, Iter {i+1}/{total_steps} | Loss: {loss.item():.6f} | {iter_speed:.4f} s/iter")

            avg_epoch_loss = np.average(epoch_losses)
            val_loss = self.vali(vali_data, vali_loader, criterion)
            all_val_losses.append(val_loss)

            print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.6f} | Vali Loss: {val_loss:.6f}")
            early_stopping(val_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            adjust_learning_rate(optimizer, epoch+1, self.args)

        best_model_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model, np.mean(all_val_losses)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.padding == 0:
                    pad = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                else:
                    pad = torch.ones((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], pad], dim=1)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                # For evaluation, convert log probabilities to probabilities if using KL divergence
                outputs = torch.exp(outputs)
                f_dim = -1 if self.args.features == 'MS' else 0

                target = batch_y[:, -self.args.pred_len:, f_dim:]
                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print("Test predictions shape:", all_preds.shape)
        mae, mse, rmse, mape, mspe = metric(all_preds, all_targets)
        print(f"Test MAE: {mae:.6f}, MSE: {mse:.6f}")
        results_dir = os.path.join('./results/', setting)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.save(os.path.join(results_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(results_dir, 'pred.npy'), all_preds)
        np.save(os.path.join(results_dir, 'true.npy'), all_targets)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            checkpoint_dir = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if self.args.padding == 0:
                    pad = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                else:
                    pad = torch.ones((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]), device=self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], pad], dim=1)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                outputs = torch.exp(outputs)
                all_preds.append(outputs.detach().cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        results_dir = os.path.join('./results/', setting)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.save(os.path.join(results_dir, 'real_prediction.npy'), all_preds)
        return