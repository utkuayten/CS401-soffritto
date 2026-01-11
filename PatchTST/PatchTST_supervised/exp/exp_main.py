from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import json
import warnings

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Informer": Informer,
            "DLinear": DLinear,
            "NLinear": NLinear,
            "Linear": Linear,
            "PatchTST": PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Keep as-is (do not change your tuned parameters)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.KLDivLoss(reduction="batchmean", log_target=False)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "Linear" in self.args.model or "TST" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if "Linear" in self.args.model or "TST" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        self.model.train()
        return float(np.average(total_loss)) if len(total_loss) else 0.0

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "Linear" in self.args.model or "TST" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                        loss = criterion(outputs, y)
                else:
                    if "Linear" in self.args.model or "TST" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # you had a 5th arg in one branch; keep your behavior
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    loss = criterion(outputs, y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / max(1, iter_count)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == "TST":
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss_val = float(np.average(train_loss)) if len(train_loss) else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch+1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss_val:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f"Updating learning rate to {scheduler.get_last_lr()[0]}")

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _ensure_dir(self, p: str):
        os.makedirs(p, exist_ok=True)

    def _save_args_json(self, folder_path: str):
        d = vars(self.args).copy()
        # argparse sets are not JSON-serializable
        for k, v in list(d.items()):
            if isinstance(v, set):
                d[k] = sorted(list(v))
        with open(os.path.join(folder_path, "args.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"), map_location=self.device)
            )

        preds = []
        trues = []
        inputx = []

        # Save visuals + also mirror arrays here for convenience
        test_folder = os.path.join("./test_results", setting)
        self._ensure_dir(test_folder)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "Linear" in self.args.model or "TST" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if "Linear" in self.args.model or "TST" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(test_folder, f"{i}.pdf"))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # Main results folder: prefer args.results_path if present
        results_root = getattr(self.args, "results_path", "./results")
        results_folder = os.path.join(results_root, setting)
        self._ensure_dir(results_folder)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}, rse:{rse}")

        # Append result summary
        with open("result.txt", "a", encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, mae:{mae}, rse:{rse}")
            f.write("\n\n")

        metrics_arr = np.array([mae, mse, rmse, mape, mspe, rse, corr], dtype=np.float64)
        metrics_json = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "rse": float(rse),
            "corr": float(corr),
            "n_test_samples": int(preds.shape[0]),
        }

        # ---- SAVE (results/<setting>/...) ----
        np.save(os.path.join(results_folder, "pred.npy"), preds)
        np.save(os.path.join(results_folder, "true.npy"), trues)
        np.save(os.path.join(results_folder, "metrics.npy"), metrics_arr)
        with open(os.path.join(results_folder, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2)
        self._save_args_json(results_folder)

        # ---- MIRROR SAVE (test_results/<setting>/...) ----
        # This removes ALL confusion about where to read arrays from.
        np.save(os.path.join(test_folder, "pred.npy"), preds)
        np.save(os.path.join(test_folder, "true.npy"), trues)
        np.save(os.path.join(test_folder, "metrics.npy"), metrics_arr)
        with open(os.path.join(test_folder, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2)
        self._save_args_json(test_folder)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "Linear" in self.args.model or "TST" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if "Linear" in self.args.model or "TST" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        results_root = getattr(self.args, "results_path", "./results")
        folder_path = os.path.join(results_root, setting)
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, "real_prediction.npy"), preds)
        self._save_args_json(folder_path)
        return