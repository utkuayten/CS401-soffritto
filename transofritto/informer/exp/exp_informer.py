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
    """Batched Soffritto LSTM teacher.

    Expects x: [B, S, input_size] and returns log-probs: [B, S, output_size].
    Architecture matches soffritto's predict_intra_cell_line.py (BiLSTM + FC + LogSoftmax),
    but implemented with batch_first=True for convenience.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # [B,S,2H]
        out = self.fc(out)             # [B,S,O]
        out = self.log_softmax(out)    # log-probs
        return out


class Exp_Informer(Exp_Basic):
    """
    Decoder input modes (args.decoding_mode):
      1) "teacher-forced": dec_inp = [Y[:label_len]] + [pad(pred_len)]
      2) "cost-aware-1" : dec_inp = [proj(X_last_label_len)] + [pad(pred_len)]
                          (uses ALL enc_in features, e.g., 9)
      3) "cost-aware-2" : dec_inp = [proj(rt2_only_from_X)] + [pad(pred_len)]
                          (uses ONLY 1 feature: rt2 / 2rt)
      4) "lstm-teacher"  : dec_inp = [proj(SoFfrittoLSTM(X_last_label_len))] + [pad(pred_len)]
                          (uses pretrained Soffritto LSTM to generate a distribution over 16 fractions)
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


        # Optional: Soffritto-LSTM teacher for decoding_mode="lstm-teacher"
        self.lstm_teacher = None
        self.lstm_teacher_out_dim = 16  # Soffritto predicts 16-fraction distribution
        if str(self.decoding_mode).strip().lower() in ("lstm-teacher", "lstm_teacher", "lstm", "teacher-lstm"):
            self._init_lstm_teacher()

        # Trainable projection 16 -> dec_in (only used if your decoder expects !=16 channels)
        if not hasattr(self.model, "dec_proj_lstm"):
            self.model.dec_proj_lstm = nn.Linear(self.lstm_teacher_out_dim, self.args.dec_in).to(self.device)

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
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
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

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
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
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
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
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                pred, _ = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
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

    

    def _init_lstm_teacher(self):
        """Load pretrained Soffritto LSTM and freeze it.

        Required args when decoding_mode=lstm-teacher:
          - args.lstm_teacher_ckpt: path to .pth/.pt state_dict
          - args.lstm_teacher_hparams_json (optional): JSON containing {num_hiddens, num_layers}
            OR args.lstm_teacher_hidden / args.lstm_teacher_layers
        The teacher takes the SAME preprocessed encoder inputs batch_x (scaled, same feature order).
        """
        ckpt = getattr(self.args, "lstm_teacher_ckpt", None)
        if ckpt is None:
            raise ValueError("decoding_mode=lstm-teacher requires --lstm_teacher_ckpt")

        # Resolve hidden/layers from json or args
        hidden = getattr(self.args, "lstm_teacher_hidden", 128)
        layers = getattr(self.args, "lstm_teacher_layers", 2)
        hp_json = getattr(self.args, "lstm_teacher_hparams_json", None)
        if hp_json:
            import json
            with open(hp_json, "r") as f:
                hp = json.load(f)
            # support both keys used in various soffritto scripts
            hidden = int(hp.get("num_hiddens", hp.get("hidden_size", hidden)))
            layers = int(hp.get("num_layers", layers))

        self.lstm_teacher = SoffrittoTeacher(
            input_size=self.args.enc_in,
            hidden_size=hidden,
            num_layers=layers,
            output_size=self.lstm_teacher_out_dim,
        ).to(self.device)

        state = torch.load(ckpt, map_location=self.device)
        # tolerate checkpoints saved as {"model_state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # strip DataParallel prefix if present
        if isinstance(state, dict):
            state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }

        missing, unexpected = self.lstm_teacher.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            print(f"[WARN] SoffrittoTeacher unexpected keys: {unexpected}")
        if len(missing) > 0:
            print(f"[WARN] SoffrittoTeacher missing keys: {missing}")

        self.lstm_teacher.eval()
        for p in self.lstm_teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _lstm_teacher_predict_probs(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Return teacher probabilities (not log) for each timestep.

        batch_x: [B, S, enc_in] (already scaled/preprocessed by Dataset_Custom)
        returns : [B, S, 16] probabilities summing to 1 on last dim
        """
        if self.lstm_teacher is None:
            self._init_lstm_teacher()

        logp = self.lstm_teacher(batch_x)       # [B,S,16] log-probs
        probs = torch.exp(logp)                 # convert to probs for decoder input / KL targets convention
        return probs

    def _build_decoder_input_lstm_teacher(self, batch_x: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
        """dec_inp = [proj(teacher_probs_last_label_len)] + [pad(pred_len)]"""
        teacher_probs = self._lstm_teacher_predict_probs(batch_x)     # [B, seq_len, 16]
        y_hist = teacher_probs[:, -label_len:, :]                     # [B, label_len, 16]

        if self.args.dec_in == self.lstm_teacher_out_dim:
            dec_hist = y_hist
        else:
            dec_hist = self.model.dec_proj_lstm(y_hist)

        dec_pad = torch.zeros(batch_x.size(0), pred_len, self.args.dec_in, device=self.device)
        dec_inp = torch.cat([dec_hist, dec_pad], dim=1)               # [B, label_len+pred_len, dec_in]
        return dec_inp
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
            # ---- to device ----
            batch_x = batch_x.float().to(self.device)          # [B, seq_len, enc_in]
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # batch_y needed for teacher-forced true + decoder history in TF
            batch_y = batch_y.float().to(self.device)

            pred_len = self.args.pred_len
            label_len = self.args.label_len

            # ---- build decoder input based on mode ----
            mode = getattr(self.args, "decode_mode", self.decoding_mode)
            mode = str(mode).strip().lower()
            if mode in ("teacher-forced", "teacher_forced", "tf", "teacher"):
                dec_inp = self._build_decoder_input_teacher_forced(batch_y, label_len, pred_len)

            elif mode in ("cost-aware-1", "cost_aware_1", "ca1", "costaware1"):
                dec_inp = self._build_decoder_input_cost_aware_1(batch_x, label_len, pred_len)

            elif mode in ("cost-aware-2", "cost_aware_2", "ca2", "costaware2"):
                dec_inp = self._build_decoder_input_cost_aware_2(batch_x, label_len, pred_len)

            elif mode in ("lstm-teacher", "lstm_teacher", "lstm", "teacher-lstm"):
                dec_inp = self._build_decoder_input_lstm_teacher(batch_x, label_len, pred_len)

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
