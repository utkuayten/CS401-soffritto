import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your models and data loaders.
  # plus other models if needed
from iTransformer.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from iTransformer.utils.metrics import metric
from iTransformer.utils.tools import EarlyStopping, adjust_learning_rate
from iTransformer.experiments.exp_basic import Exp_Basic
# Experiment class for iTransformer
class Exp_iTransformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_iTransformer, self).__init__(args)

    def _build_model(self):
        # Build your iTransformer model with parameters from args.
        model = self.model_dict[self.args.model].Model(self.args).float()


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        # Mapping from dataset name to dataset class
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
        # Choose dataset class based on the specified data argument and flag.
        if flag == 'pred':
            Data = Dataset_Pred
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
        else:
            Data = data_dict.get(args.data, Dataset_Custom)
            shuffle_flag = True if flag == 'train' else False
            drop_last = True if flag in ['train', 'test'] else False
            batch_size = args.batch_size
            freq = args.freq

        # Determine whether to time-encode depending on embedding type.
        timeenc = 0 if args.embed != 'timeF' else 1

        dataset = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
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
        # Using KLDivLoss for log-softmax outputs. Adjust if needed.
        return nn.KLDivLoss(reduction='batchmean')

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        # Load data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create directory for checkpoints if it does not exist.
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # For mixed precision training if enabled.
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        validation_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Iter {i+1} | Loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"Speed: {speed:.4f}s/iter, Estimated left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch {epoch+1} completed in {time.time() - epoch_time:.2f} seconds")
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            validation_loss.append(vali_loss)

            print(f"Epoch: {epoch+1}, Train Loss: {train_loss_avg:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model, np.mean(validation_loss)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print("Raw test shapes:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("Reshaped test shapes:", preds.shape, trues.shape)

        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"Test MAE: {mae:.4f}, MSE: {mse:.4f}")
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, _ = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        return

    def _process_one_batch(self, dataset_obj, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # Move data to device and convert to float
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Create decoder input by concatenating known target values and zero or one padding.
        if self.args.padding == 0:
            dec_inp = torch.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1])).float()
        else:
            dec_inp = torch.ones((batch_y.shape[0], self.args.pred_len, batch_y.shape[-1])).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # Forward pass. Use AMP autocast if enabled.
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_obj.inverse_transform(outputs)

        # Adjust target to the proper shape: here f_dim is selected based on feature type.
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # Apply log softmax on the output if required by the KLDivLoss.
        outputs = F.log_softmax(outputs, dim=-1)
        #print(outputs[0])
        #return null
        return outputs, batch_y