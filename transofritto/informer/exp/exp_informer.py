import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from transformers import get_linear_schedule_with_warmup
import warnings

from transofritto.informer.data_loader.data_loader import Dataset_Custom
from transofritto.informer.exp.exp_basic import Exp_Basic
from transofritto.informer.models.model import Informer, InformerStack
from transofritto.informer.utils.metrics import metric
from transofritto.informer.utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")

kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")

class SoffrittoTeacher(nn.Module):
    """Minimal Soffritto-style BiLSTM teacher.

    Input:  x   [B, S, 9]   (UNSCALED features in SOFFRITTO_TEACHER_COLS order)
    Output: logp[B, S, 16]  (LogSoftmax over fractions)
    """
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out)
        return self.logsoftmax(out)



class Exp_Informer(Exp_Basic):
    """
    Decoder input modes (args.decoding_mode):
      1) "teacher-forced": dec_inp = [Y[:label_len]] + [pad(pred_len)]
      2) "cost-aware-1" : dec_inp = [proj(X_last_label_len)] + [pad(pred_len)]
                          (uses ALL enc_in features, e.g., 9)
      3) "cost-aware-2" : dec_inp = [proj(rt2_only_from_X)] + [pad(pred_len)]
                          (uses ONLY 1 feature: rt2 / 2rt)
    """

    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.args = args  # ✅ Store args in self

        # will be filled from dataset (index of rt2 feature inside X)
        self.rt2_idx = None

        # default decoding mode
        # accepted: "teacher-forced", "cost-aware-1", "cost-aware-2"
        self.decoding_mode = getattr(self.args, "decoding_mode", "teacher-forced")

        # Trainable projection enc_in -> dec_in (for cost-aware-1 if enc_in != dec_in)
        # Attach to model so optimizer(self.model.parameters()) sees it.
        if not hasattr(self.model, "dec_proj"):
            self.model.dec_proj = nn.Linear(self.args.enc_in, self.args.dec_in).to(self.device)

        # Trainable projection 1 -> dec_in (for cost-aware-2)
        if not hasattr(self.model, "dec_proj_rt2"):
            self.model.dec_proj_rt2 = nn.Linear(1, self.args.dec_in).to(self.device)


        # Trainable projection 16 -> dec_in (for lstm-teacher if dec_in != 16)
        if not hasattr(self.model, "dec_proj_lstm"):
            self.model.dec_proj_lstm = nn.Linear(16, self.args.dec_in).to(self.device)

        # Optional: Soffritto LSTM teacher for decoder history
        self.lstm_teacher = None
        mode0 = str(getattr(self.args, "decoding_mode", getattr(self.args, "decode_mode", self.decoding_mode))).strip().lower()
        if mode0 in ("lstm-teacher", "lstm_teacher", "teacher-lstm", "lstm"):
            self._init_lstm_teacher()
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _build_model(self):
        model_dict = {"informer": Informer, "informerstack": InformerStack}

        if self.args.model == "informer" or self.args.model == "informerstack":
            _ = self.args.e_layers if self.args.model == "informer" else self.args.s_layers

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
                use_wavelet=getattr(self.args, "use_wavelet", False),
                wavelet=getattr(self.args, "wavelet_name", "db4"),
                levels=getattr(self.args, "wavelet_levels", 1),
                keep_original=getattr(self.args, "keep_original", True),
                wavelet_where=getattr(self.args, "wavelet_where", "model"),
                selected_cols=self.args.selected_cols,
            ).float()
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


        # ---------------------------
        # Soffritto LSTM teacher init
        # ---------------------------
    def _init_lstm_teacher(self):
        ckpt_path = getattr(self.args, "lstm_teacher_ckpt", None)
        if not ckpt_path:
            raise ValueError("decoding_mode=lstm-teacher requires --lstm_teacher_ckpt pointing to the trained Soffritto LSTM checkpoint.")

        hidden = int(getattr(self.args, "lstm_teacher_hidden", 128))
        layers = int(getattr(self.args, "lstm_teacher_layers", 2))

        teacher = SoffrittoTeacher(input_dim=9, hidden_dim=hidden, num_layers=layers, num_classes=16).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        # accept multiple checkpoint formats
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt

        # strip DataParallel prefix if present
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v

        missing, unexpected = teacher.load_state_dict(new_state, strict=False)
        # Freeze teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        self.lstm_teacher = teacher
        if missing:
            print(f"[WARN] LSTM teacher missing keys: {missing}")
        if unexpected:
            print(f"[WARN] LSTM teacher unexpected keys: {unexpected}")

    @torch.no_grad()
    def _lstm_teacher_predict_logp(self, batch_x_teacher: torch.Tensor) -> torch.Tensor:
        if self.lstm_teacher is None:
            raise RuntimeError("LSTM teacher not initialized. Check decoding_mode and lstm_teacher_ckpt.")
        return self.lstm_teacher(batch_x_teacher)

    def _build_decoder_input_lstm_teacher(self, batch_x_teacher: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        """Build decoder input using Soffritto teacher predictions as history.

        batch_x_teacher: [B, seq_len, 9] UN-SCALED, Soffritto-ordered features
        returns dec_inp: [B, label_len+pred_len, dec_in]
        """
        teacher_logp = self._lstm_teacher_predict_logp(batch_x_teacher)      # [B, seq_len, 16] log-prob
        teacher_probs = torch.exp(teacher_logp)                               # [B, seq_len, 16] prob

        y_hist = teacher_probs[:, -label_len:, :]                             # [B, label_len, 16]

        if self.args.dec_in == 16:
            dec_hist = y_hist
        else:
            dec_hist = self.model.dec_proj_lstm(y_hist)                       # [B, label_len, dec_in]

        dec_pad = torch.zeros(batch_x_teacher.size(0), pred_len, self.args.dec_in, device=self.device)
        return torch.cat([dec_hist, dec_pad], dim=1)
    def _get_data(self, flag):
        args = self.args
        data_dict = {"custom": Dataset_Custom}
        Data = data_dict[self.args.data]

        timeenc = 0 if args.embed != "timeF" else 1
        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

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
            train_chroms=args.train_chroms,
            val_chroms=args.val_chroms,
            test_chroms=args.test_chroms,
            selected_cols=args.selected_cols,
        )

        # Read rt2 index once (must exist in your Dataset_Custom)
        if self.rt2_idx is None:
            # expected: data_set.rt2_idx exists and points into X feature dimension (enc_in)
            self.rt2_idx = getattr(data_set, "rt2_idx", None)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _select_scheduler(self, optimizer, train_loader):
        num_training_steps = len(train_loader) * self.args.train_epochs
        num_warmup_steps = int(0.2 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def _select_criterion(self):
        return nn.KLDivLoss(reduction="batchmean", log_target=False)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x_teacher = batch[4] if len(batch) > 4 else None
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_teacher)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())

        avg_loss = sum(total_loss) / max(1, len(total_loss))
        print(f"Validation KL div loss: {avg_loss:.6f}")
        self.model.train()
        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        self.model.to(self.device)
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # scheduler = self._select_scheduler(model_optim, train_loader)
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        validation_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x_teacher = batch[4] if len(batch) > 4 else None
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,
                    batch_x_teacher=batch_x_teacher
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / max(1, iter_count)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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
                    # scheduler.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss_avg = float(np.average(train_loss)) if train_loss else float("inf")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            validation_loss.append(vali_loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Keep original behavior (if your tools.py supports chosen lradj)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model, min(validation_loss) if validation_loss else float("inf")

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")
        self.model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x_teacher = batch[4] if len(batch) > 4 else None
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_teacher)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)

        folder_path = os.path.join(self.args.results_path, setting)
        os.makedirs(folder_path, exist_ok=True)

        results = metric(preds, trues)

        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        np.savez(os.path.join(folder_path, "metrics.npz"), **results)

        with open(os.path.join(folder_path, "metrics.txt"), "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.6f}\n")

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in pred_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x_teacher = batch[4] if len(batch) > 4 else None
                pred, _ = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_teacher)
                preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        folder_path = os.path.join("./results", setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, "real_prediction.npy"), preds)
        return

    def _build_decoder_input_teacher_forced(self, batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        """
        dec_inp = [batch_y[:label_len]] + [pad(pred_len)]
        """
        padding_mode = getattr(self.args, "padding", 0)  # 0 zeros, 1 ones
        if padding_mode == 1:
            dec_pad = torch.ones_like(batch_y[:, -pred_len:, :]).float()
        else:
            dec_pad = torch.zeros_like(batch_y[:, -pred_len:, :]).float()

        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_pad], dim=1).float().to(self.device)
        return dec_inp

    def _build_decoder_input_cost_aware_1(self, batch_x: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        """
        dec_inp = [proj(X_last_label_len)] + [pad(pred_len)]
        Uses ALL enc_in features (e.g., 9). Project enc_in -> dec_in if needed.
        """
        x_hist = batch_x[:, -label_len:, :]  # [B, label_len, enc_in]

        # project to dec_in (works even if enc_in==dec_in)
        if self.args.enc_in == self.args.dec_in:
            dec_hist = x_hist
        else:
            dec_hist = self.model.dec_proj(x_hist)  # [B, label_len, dec_in]

        dec_pad = torch.zeros(batch_x.size(0), pred_len, self.args.dec_in, device=self.device)
        dec_inp = torch.cat([dec_hist, dec_pad], dim=1)  # [B, label_len+pred_len, dec_in]
        return dec_inp

    def _build_decoder_input_cost_aware_2(self, batch_x: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        """
        dec_inp = [proj(rt2_only_from_X)] + [pad(pred_len)]
        Uses ONLY 1 feature (rt2 / 2rt) inside the 9 enc_in features.
        """
        if self.rt2_idx is None:
            raise RuntimeError(
                "rt2_idx is None. Your Dataset_Custom must define data_set.rt2_idx "
                "(index of rt2/2rt feature inside encoder features)."
            )

        x_rt2 = batch_x[:, -label_len:, self.rt2_idx:self.rt2_idx + 1]  # [B, label_len, 1]
        dec_hist = self.model.dec_proj_rt2(x_rt2)  # [B, label_len, dec_in]

        dec_pad = torch.zeros(batch_x.size(0), pred_len, self.args.dec_in, device=self.device)
        dec_inp = torch.cat([dec_hist, dec_pad], dim=1)  # [B, label_len+pred_len, dec_in]
        return dec_inp

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_teacher=None):
        # ---- to device ----
        batch_x = batch_x.float().to(self.device)          # [B, seq_len, enc_in]
        # Optional: unscaled 9-dim features for Soffritto LSTM teacher
        if batch_x_teacher is not None:
            batch_x_teacher = batch_x_teacher.float().to(self.device)  # [B, seq_len, 9]
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # batch_y needed for teacher-forced true + decoder history in TF
        batch_y = batch_y.float().to(self.device)

        pred_len = self.args.pred_len
        label_len = self.args.label_len

        # ---- build decoder input based on mode ----
        mode = getattr(self.args, "decoding_mode", getattr(self.args, "decode_mode", self.decoding_mode))
        mode = str(mode).strip().lower()
        if mode in ("teacher-forced", "teacher_forced", "tf", "teacher"):
            dec_inp = self._build_decoder_input_teacher_forced(batch_y, label_len, pred_len)

        elif mode in ("cost-aware-1", "cost_aware_1", "ca1", "costaware1"):
            dec_inp = self._build_decoder_input_cost_aware_1(batch_x, label_len, pred_len)

        elif mode in ("cost-aware-2", "cost_aware_2", "ca2", "costaware2"):
            dec_inp = self._build_decoder_input_cost_aware_2(batch_x, label_len, pred_len)

        elif mode in ("lstm-teacher", "lstm_teacher", "teacher-lstm", "lstm"):
            if batch_x_teacher is None:
                raise ValueError("decoding_mode=lstm-teacher requires Dataset_Custom to return seq_x_teacher. Use data_loader_fixed.py changes.")
            dec_inp = self._build_decoder_input_lstm_teacher(batch_x_teacher, label_len, pred_len)

        else:
            raise ValueError(
                f"Unknown decoding_mode={mode!r}. "
                f"Use one of: teacher-forced, cost-aware-1, cost-aware-2, lstm-teacher"
            )
        # ---- forward ----
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if getattr(self.args, "output_attention", False):
            outputs = outputs[0]

        if getattr(self.args, "inverse", False):
            outputs = dataset_object.inverse_transform(outputs)

        # ---- slice to pred horizon (so KL shapes match) ----
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -pred_len:, f_dim:]          # [B, pred_len, ?]

        true = batch_y[:, -pred_len:, f_dim:]             # [B, pred_len, ?]
        return outputs, true