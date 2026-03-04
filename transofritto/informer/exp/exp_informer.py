import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import json
from transformers import get_linear_schedule_with_warmup
import warnings

from transofritto.informer.data_loader.data_loader import Dataset_Custom
from transofritto.informer.exp.exp_basic import Exp_Basic
from transofritto.informer.models.model import Informer, InformerStack
from transofritto.informer.utils.metrics import metric
from transofritto.informer.utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")

kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")


class SoffrittoLSTM(nn.Module):
    """Soffritto BiLSTM teacher (frozen) to generate decoder history for cost-aware-3.

    This implementation uses batch_first=True so it consumes [B, T, F].
    It outputs log-probabilities over 16 RT fractions at each timestep: [B, T, 16].
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.hidden = None  # (h, c)

    def reset_hidden(self, batch_size: int, device: torch.device):
        num_directions = 2
        h0 = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, device=device)
        self.hidden = (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden is None or self.hidden[0].size(1) != x.size(0):
            self.reset_hidden(batch_size=x.size(0), device=x.device)

        out, self.hidden = self.lstm(x, self.hidden)

        # Detach hidden to avoid graph growth
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        out = self.fc(out)
        out = self.log_softmax(out)
        return out


class Exp_Informer(Exp_Basic):
    """
    Decoder input modes (args.decoding_mode):
      1) "teacher-forced": dec_inp = [Y[:label_len]] + [pad(pred_len)]
      2) "cost-aware-1" : dec_inp = [proj(X_informer_last_label_len)] + [pad(pred_len)]
                          (uses Informer feature pipeline)
      3) "cost-aware-2" : dec_inp = [proj(rt2_only_from_X_informer)] + [pad(pred_len)]
                          (uses ONLY 1 feature: rt2/2-stage from Informer pipeline)
      4) "cost-aware-3" : dec_inp = [proj(LSTM_pred_probs_last_label_len)] + [pad(pred_len)]
                          (LSTM uses its OWN feature pipeline from Dataset_Custom, not Informer X)
    """

    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.args = args

        # will be filled from dataset: index of rt2 within Informer input features
        self.rt2_idx = None

        # accepted: teacher-forced, cost-aware-1, cost-aware-2, cost-aware-3
        self.decoding_mode = getattr(self.args, "decoding_mode", "teacher-forced")

        # trainable projections are attached to model so optimizer sees them
        if not hasattr(self.model, "dec_proj"):
            self.model.dec_proj = nn.Linear(self.args.enc_in, self.args.dec_in).to(self.device)
        if not hasattr(self.model, "dec_proj_rt2"):
            self.model.dec_proj_rt2 = nn.Linear(1, self.args.dec_in).to(self.device)
        if not hasattr(self.model, "dec_proj_lstm"):
            self.model.dec_proj_lstm = nn.Linear(16, self.args.dec_in).to(self.device)

        # cost-aware-3 teacher
        self.lstm_teacher = None
        self._lstm_teacher_loaded = False

        mode0 = str(getattr(self.args, "decoding_mode", self.decoding_mode)).strip().lower()
        if mode0 in ("cost-aware-3", "cost_aware_3", "ca3", "costaware3"):
            self._load_lstm_teacher()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack
        }

        if self.args.model == "informer" or self.args.model == "informerstack":
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
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == "test":
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Dataset_Custom(
            root_path=args.root_path,
            train_chroms=args.train_chroms,
            test_chroms=args.test_chroms,
            val_chroms=args.val_chroms,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            inverse=args.inverse,
            timeenc=1 if args.embed == "timeF" else 0,
            freq=freq,
            selected_cols=args.selected_cols,
            # NEW: separate LSTM pipeline
            lstm_selected_cols=getattr(args, "lstm_selected_cols", None),
            lstm_scale=getattr(args, "lstm_scale", True),
            rt2_col_name=getattr(args, "rt2_col", "2-stage"),
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        # dataset provides rt2 index
        if hasattr(data_set, "rt2_idx"):
            self.rt2_idx = data_set.rt2_idx

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return kl_criterion

    def _maybe_adjust_lr(self, optimizer, scheduler, epoch):
        """
        Robust wrapper for adjust_learning_rate with differing signatures.
        Supports common variants:
          - adjust_learning_rate(optimizer, epoch, args)
          - adjust_learning_rate(optimizer, scheduler, epoch, args)  (some forks)
          - adjust_learning_rate(optimizer, epoch)                   (rare)
        """
        import inspect
        sig = inspect.signature(adjust_learning_rate)
        n = len(sig.parameters)

        if n == 4:
            # (optimizer, scheduler, epoch, args)
            return adjust_learning_rate(optimizer, scheduler, epoch, self.args)
        elif n == 3:
            # (optimizer, epoch, args)
            return adjust_learning_rate(optimizer, epoch, self.args)
        elif n == 2:
            # (optimizer, epoch)
            return adjust_learning_rate(optimizer, epoch)
        else:
            raise TypeError(f"Unsupported adjust_learning_rate signature: {sig}")

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # NEW: dataloader now returns 5-tuple
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm = batch
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm
                )
                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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

        warmup_steps = int(train_steps * self.args.train_epochs * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            model_optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps * self.args.train_epochs
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1

                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm = batch

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                scheduler.step()

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            self._maybe_adjust_lr(model_optim, scheduler, epoch + 1)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model, early_stopping.best_score

    def test(self, setting):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm = batch
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm
                )
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        m = metric(preds, trues)
        # FIX: metric() may return more than 5 values depending on implementation.
        if isinstance(m, (list, tuple)):
            if len(m) >= 5:
                mae, mse, rmse, mape, mspe = m[:5]
            else:
                raise ValueError(f"metric() returned {len(m)} values, expected >= 5")
        elif isinstance(m, dict):
            mae = m.get("mae")
            mse = m.get("mse")
            rmse = m.get("rmse")
            mape = m.get("mape")
            mspe = m.get("mspe")
            if any(v is None for v in (mae, mse, rmse, mape, mspe)):
                raise ValueError(f"metric() returned dict without expected keys: {m.keys()}")
        else:
            raise TypeError(f"Unexpected metric() return type: {type(m)}")

        print("mse:{}, mae:{}".format(mse, mae))
        return

    # ---------------- cost-aware-3: teacher loader + decoder input builder ----------------
    def _load_lstm_teacher(self):
        if self._lstm_teacher_loaded:
            return

        ckpt_path = getattr(self.args, "lstm_model_path", None)
        if ckpt_path is None:
            raise ValueError(
                "cost-aware-3 requires a pretrained LSTM teacher checkpoint. "
                "Provide --lstm_model_path pointing to the .pth saved by train_intra_cell_line.py"
            )

        hidden_size = getattr(self.args, "lstm_hidden_size", None)
        num_layers = getattr(self.args, "lstm_num_layers", None)
        hjson = getattr(self.args, "lstm_hyperparameter_file", None)

        if (hidden_size is None or num_layers is None) and hjson is not None:
            with open(hjson, "r") as f:
                d = json.load(f)
            hidden_size = hidden_size if hidden_size is not None else d.get("hidden_size")
            num_layers = num_layers if num_layers is not None else d.get("num_layers")

        if hidden_size is None or num_layers is None:
            raise ValueError(
                "Provide LSTM hyperparameters via --lstm_hidden_size and --lstm_num_layers "
                "or via --lstm_hyperparameter_file JSON."
            )

        # IMPORTANT: teacher input_size must match LSTM pipeline feature dim.
        # We may not know it at init time; we rebuild on first batch if needed.
        input_size = int(getattr(self.args, "lstm_input_size", 0) or 0)
        if input_size <= 0:
            input_size = 1  # placeholder (rebuilt later)

        self.lstm_teacher = SoffrittoLSTM(
            input_size=input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            output_size=16,
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)

        try:
            self.lstm_teacher.load_state_dict(state, strict=True)
            self._teacher_state_dict = None
        except Exception:
            # Defer strict load until we know true input size (rebuild in _ensure_teacher_input_size).
            self._teacher_state_dict = state

        self.lstm_teacher.eval()
        for p in self.lstm_teacher.parameters():
            p.requires_grad_(False)

        self._lstm_teacher_loaded = True

    def _ensure_teacher_input_size(self, x_lstm: torch.Tensor):
        """If teacher was constructed with placeholder input_size, rebuild it with correct size and load weights."""
        if getattr(self, "_teacher_state_dict", None) is None:
            return
        true_in = x_lstm.size(-1)
        hidden_size = self.lstm_teacher.hidden_size
        num_layers = self.lstm_teacher.num_layers

        self.lstm_teacher = SoffrittoLSTM(
            input_size=true_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=16,
        ).to(self.device)
        self.lstm_teacher.load_state_dict(self._teacher_state_dict, strict=True)
        self._teacher_state_dict = None

        self.lstm_teacher.eval()
        for p in self.lstm_teacher.parameters():
            p.requires_grad_(False)

    def _lstm_teacher_predict_probs(self, batch_x_lstm: torch.Tensor) -> torch.Tensor:
        if not self._lstm_teacher_loaded:
            self._load_lstm_teacher()

        self._ensure_teacher_input_size(batch_x_lstm)

        with torch.no_grad():
            self.lstm_teacher.reset_hidden(batch_size=batch_x_lstm.size(0), device=batch_x_lstm.device)
            logp = self.lstm_teacher(batch_x_lstm)            # [B, T, 16] log-probs
            probs = torch.exp(logp).clamp_min(1e-12)          # [B, T, 16] probs
        return probs

    def _build_decoder_input_teacher_forced(self, batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        padding_mode = getattr(self.args, "padding", 0)  # 0 zeros, 1 ones
        if padding_mode == 1:
            dec_pad = torch.ones_like(batch_y[:, -pred_len:, :]).float()
        else:
            dec_pad = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_pad], dim=1).float().to(self.device)
        return dec_inp

    def _build_decoder_input_cost_aware_1(self, batch_x_informer: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        x_hist = batch_x_informer[:, -label_len:, :]  # [B, label_len, enc_in]
        if self.args.enc_in == self.args.dec_in:
            dec_hist = x_hist
        else:
            dec_hist = self.model.dec_proj(x_hist)
        dec_pad = torch.zeros(batch_x_informer.size(0), pred_len, self.args.dec_in, device=self.device)
        return torch.cat([dec_hist, dec_pad], dim=1)

    def _build_decoder_input_cost_aware_2(self, batch_x_informer: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        if self.rt2_idx is None:
            raise RuntimeError(
                "rt2_idx is None. Ensure Dataset_Custom defines rt2_idx (rt2_col_name index in input features)."
            )
        x_rt2 = batch_x_informer[:, -label_len:, self.rt2_idx:self.rt2_idx + 1]  # [B, label_len, 1]
        dec_hist = self.model.dec_proj_rt2(x_rt2)
        dec_pad = torch.zeros(batch_x_informer.size(0), pred_len, self.args.dec_in, device=self.device)
        return torch.cat([dec_hist, dec_pad], dim=1)

    def _build_decoder_input_cost_aware_3(self, batch_x_lstm: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        teacher_probs = self._lstm_teacher_predict_probs(batch_x_lstm)     # [B, seq_len, 16]
        y_hist = teacher_probs[:, -label_len:, :]                          # [B, label_len, 16]

        if self.args.dec_in == 16:
            dec_hist = y_hist
        else:
            dec_hist = self.model.dec_proj_lstm(y_hist)

        dec_pad = torch.zeros(batch_x_lstm.size(0), pred_len, self.args.dec_in, device=self.device)
        return torch.cat([dec_hist, dec_pad], dim=1)

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_lstm):
        # ---- RAW inputs from dataloader ----
        batch_x_informer = batch_x.float().to(self.device)            # Informer input pipeline
        batch_x_lstm = batch_x_lstm.float().to(self.device)           # LSTM input pipeline (independent)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        pred_len = self.args.pred_len
        label_len = self.args.label_len

        mode = str(getattr(self.args, "decoding_mode", self.decoding_mode)).strip().lower()

        if mode in ("teacher-forced", "teacher_forced", "tf", "teacher"):
            dec_inp = self._build_decoder_input_teacher_forced(batch_y, label_len, pred_len)
        elif mode in ("cost-aware-1", "cost_aware_1", "ca1", "costaware1"):
            dec_inp = self._build_decoder_input_cost_aware_1(batch_x_informer, label_len, pred_len)
        elif mode in ("cost-aware-2", "cost_aware_2", "ca2", "costaware2"):
            dec_inp = self._build_decoder_input_cost_aware_2(batch_x_informer, label_len, pred_len)
        elif mode in ("cost-aware-3", "cost_aware_3", "ca3", "costaware3"):
            # CRITICAL: teacher consumes ONLY batch_x_lstm (its own pipeline), not Informer encoder outputs
            dec_inp = self._build_decoder_input_cost_aware_3(batch_x_lstm, label_len, pred_len)
        else:
            raise ValueError(
                f"Unknown decoding_mode={mode!r}. Use one of: teacher-forced, cost-aware-1, cost-aware-2, cost-aware-3"
            )

        outputs = self.model(batch_x_informer, batch_x_mark, dec_inp, batch_y_mark)
        if getattr(self.args, "output_attention", False):
            outputs = outputs[0]

        if getattr(self.args, "inverse", False):
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -pred_len:, f_dim:]
        true = batch_y[:, -pred_len:, f_dim:]
        return outputs, true