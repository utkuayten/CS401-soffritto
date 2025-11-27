import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
from transformers import get_linear_schedule_with_warmup
import warnings

from transofritto.informer.data_loader.data_loader import Dataset_Custom
from transofritto.informer.exp.exp_basic import Exp_Basic
from transofritto.informer.models.model import Informer, InformerStack
from transofritto.informer.utils.metrics import metric
from transofritto.informer.utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.args = args  # ✅ Store args in self
        self.rt2_idx = None        # will be filled from dataset


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers

            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
                use_wavelet=getattr(self.args, 'use_wavelet', False),
                wavelet=getattr(self.args, 'wavelet_name', 'db4'),
                levels=getattr(self.args, 'wavelet_levels', 1),
                keep_original=getattr(self.args, 'keep_original', True),
                wavelet_where=getattr(self.args, 'wavelet_where', 'model'),
                selected_cols = self.args.selected_cols
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]

        timeenc = 0 if args.embed != 'timeF' else 1
        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq = args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            train_chroms = args.train_chroms,
            val_chroms = args.val_chroms,
            test_chroms = args.test_chroms,
            selected_cols = args.selected_cols
        )

        # we only need to read this once, columns are same for all splits
        if self.rt2_idx is None:
            self.rt2_idx = data_set.rt2_idx

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_scheduler(self, optimizer, train_loader):
        num_training_steps = len(train_loader) * self.args.train_epochs
        num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

    def _select_criterion(self):
        criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())

        avg_loss = sum(total_loss) / len(total_loss)
        print(f'Validation KL div loss: {avg_loss:.6f}')

        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model.to(self.device)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # scheduler = self._select_scheduler(model_optim, train_loader)
        criterion = self._select_criterion()

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
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    model_optim.step()
                    #scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            validation_loss.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, min(validation_loss)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('test shape:', preds.shape, trues.shape)

        folder_path = os.path.join(self.args.results_path, setting)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results = metric(preds, trues)

        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        np.savez(os.path.join(folder_path, 'metrics.npz'), **results)

        with open(os.path.join(folder_path, 'metrics.txt'), 'w') as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.6f}\n")

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x      = batch_x.float().to(self.device)
        batch_y      = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        B      = batch_y.shape[0]
        L_dec  = self.args.label_len + self.args.pred_len

        # 1) Extract last L_dec timesteps of 2-fraction RT from encoder inputs
        # batch_x: [B, seq_len, enc_in]
        # self.rt2_idx: index of 2-fraction feature in enc_in dimension
        rt2 = batch_x[:, -L_dec:, self.rt2_idx:self.rt2_idx+1]     # [B, L_dec, 1]

        # 2) Project 1-dim → dec_in (16)
        rt2_proj = self.model.rt2_to_dec(rt2)                      # [B, L_dec, dec_in]

        # 3) Use this as decoder input
        dec_inp = rt2_proj                                         # [B, L_dec, dec_in]

        # 4) Forward model
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim   = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y