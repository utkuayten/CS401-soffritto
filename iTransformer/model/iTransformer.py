import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

try:
    import pywt
except ImportError:
    pywt = None  # guarded below


class WaveletTokenizer(nn.Module):
    """
    Stationary Wavelet Transform over time per feature.
    Handles arbitrary sequence length by padding to a multiple of 2**levels,
    then truncates back to the original length.
    (B, S, F) -> (B, S, F')  where F' = F*(1+2L) if keep_original else F*(2L)
    """
    def __init__(self, wavelet='db4', levels=1, keep_original=True, pad_mode='wrap'):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.keep_original = keep_original
        self.pad_mode = pad_mode  # 'wrap' ~ periodic extension works well for SWT

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if pywt is None:
            raise RuntimeError("PyWavelets not installed. Run: pip install PyWavelets")

        B, S, F = x.shape
        dev, dtype = x.device, x.dtype
        mult_needed = 1 << self.levels  # 2**levels
        pad_len = (-S) % mult_needed     # 0 if already divisible

        x_np = x.detach().cpu().numpy()
        out_batches = []

        for b in range(B):
            parts = [x_np[b]] if self.keep_original else []  # (S, F) original (no pad)
            feat_parts = []

            for f in range(F):
                vec = x_np[b, :, f]  # (S,)
                if pad_len:
                    # pad at the tail to reach multiple of 2**levels
                    vec = np.pad(vec, (0, pad_len), mode=self.pad_mode)
                # vec length is S_pad = S + pad_len, now divisible by 2**levels
                coeffs = pywt.swt(vec, wavelet=self.wavelet, level=self.levels)
                # coeffs = [(cA1,cD1), (cA2,cD2), ...] each of length S_pad
                cAs = [cA for (cA, cD) in coeffs]
                cDs = [cD for (cA, cD) in coeffs]
                feat = np.stack(cAs + cDs, axis=1)  # (S_pad, 2L)
                if pad_len:
                    feat = feat[:S, :]              # truncate back to original S
                feat_parts.append(feat)

            feat_parts = np.concatenate(feat_parts, axis=1)    # (S, F*2L)
            parts.append(feat_parts)
            out_batches.append(torch.tensor(np.concatenate(parts, axis=1), dtype=dtype))

        return torch.stack(out_batches, dim=0).to(dev)          # (B, S, F')


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding (inverted: B L N -> B N E)
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Project to prediction horizon (token-wise): B N E -> B N S
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Final head maps token-dim (N at runtime) -> c_out
        self.c_out = configs.c_out
        self.feature_projection = nn.LazyLinear(self.c_out)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # ---- Wavelet configuration (matches the Informer example semantics) ----
        self.use_wavelet = bool(getattr(configs, 'use_wavelet', False))
        self.wavelet_where = getattr(configs, 'wavelet_where', 'model')
        self.wavelet = getattr(configs, 'wavelet_name', 'db4')
        self.levels = int(getattr(configs, 'wavelet_levels', 1))
        self.keep_original = bool(getattr(configs, 'keep_original', True))

        if self.use_wavelet:
            if pywt is None:
                raise RuntimeError("use_wavelet=True but PyWavelets is not installed. Run: pip install PyWavelets")

            # For SWT, per original feature you get 2*levels bands (cA_l and cD_l).
            # If keep_original=True, total multiplier = (1 + 2*levels), else = (2*levels).
            self.mult = (1 + 2 * self.levels) if self.keep_original else (2 * self.levels)

            # Tokenizers (no learnable params)
            self.wavetok_enc = WaveletTokenizer(
                wavelet=self.wavelet, levels=self.levels, keep_original=self.keep_original, pad_mode='wrap'
            )
            self.wavetok_dec = WaveletTokenizer(
                wavelet=self.wavelet, levels=self.levels, keep_original=self.keep_original, pad_mode='wrap'
            )

            if self.wavelet_where == "model":
                # Project expanded channels back to original dims expected by downstream embedding
                self.wproj_enc = nn.Linear(configs.enc_in * self.mult, configs.enc_in)
                self.wproj_dec = nn.Linear(configs.dec_in * self.mult, configs.dec_in)
                print(
                    f"[WAVELET/iTransformer] ON | where=model | wavelet={self.wavelet} | "
                    f"L={self.levels} | keep_original={self.keep_original} | mult=x{self.mult}"
                )
            elif self.wavelet_where == "dataset":
                # Apply tokenizer here too (to mirror the Informer example behavior),
                # but DO NOT project back; allow embedding to consume expanded token dim.
                self.enc_in_expanded = configs.enc_in * self.mult
                self.dec_in_expanded = configs.dec_in * self.mult
                print(
                    f"[WAVELET/iTransformer] ON | where=dataset | wavelet={self.wavelet} | "
                    f"L={self.levels} | keep_original={self.keep_original} | mult=x{self.mult} | "
                    f"enc_in_expanded={self.enc_in_expanded} | dec_in_expanded={self.dec_in_expanded}"
                )
            else:
                raise ValueError(f"Invalid wavelet_where='{self.wavelet_where}', must be 'model' or 'dataset'.")
        else:
            print("[WAVELET/iTransformer] OFF")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ---- Wavelet FIRST (so it actually affects the forward) ----
        if self.use_wavelet and self.wavelet_where == "model":
            wx_enc = self.wavetok_enc(x_enc)            # (B, S, enc_in * mult)
            wx_dec = self.wavetok_dec(x_dec)            # (B, S, dec_in * mult)
            x_enc = self.wproj_enc(wx_enc)              # (B, S, enc_in)
            x_dec = self.wproj_dec(wx_dec)              # (B, S, dec_in)

        if self.use_wavelet and self.wavelet_where == "dataset":
            # Mirror Informer example: apply tokenizer here as well (no projection back)
            x_enc = self.wavetok_enc(x_enc)             # (B, S, enc_in * mult or expanded already)
            x_dec = self.wavetok_dec(x_dec)             # (B, S, dec_in * mult or expanded already)

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # x_enc: B L N   -> embedding expects B L N, produces B N E (inverted)
        _, _, N = x_enc.shape  # token dimension at runtime
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # B N E

        # Project to future steps and restore token axis: B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # B S N

        if self.use_norm:
            # De-normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Final: map tokens (N) -> c_out  (lazy head adapts to actual N at first call)
        dec_out = self.feature_projection(dec_out)  # (B, S, c_out)
        dec_out = self.log_softmax(dec_out)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]