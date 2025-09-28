import torch
import torch.nn as nn
import torch.nn.functional as F

from informer.models.attn import AttentionLayer, ProbAttention, FullAttention
from informer.models.decoder import DecoderLayer, Decoder
from informer.models.embed import DataEmbedding
from informer.models.encoder import EncoderLayer, Encoder, EncoderStack, ConvLayer
import numpy as np

try:
    import pywt
except ImportError:
    pywt = None  # guarded below

import torch
import torch.nn as nn
import torch.nn.functional as F

from informer.models.attn import AttentionLayer, ProbAttention, FullAttention
from informer.models.decoder import DecoderLayer, Decoder
from informer.models.embed import DataEmbedding
from informer.models.encoder import EncoderLayer, Encoder, ConvLayer

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



class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='geno', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'),
                 # NEW:
                 use_wavelet=False, wavelet='db4', levels=1, keep_original=True,
                 wavelet_where='model'):
        super().__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        # --- Wavelet-in-model toggle ---
        self.use_wavelet = bool(use_wavelet) and (wavelet_where == 'model')
        if self.use_wavelet:
            mult = (1 + 2 * levels) if keep_original else (2 * levels)
            self.wavetok_enc = WaveletTokenizer(wavelet=wavelet, levels=levels, keep_original=keep_original)
            self.wavetok_dec = WaveletTokenizer(wavelet=wavelet, levels=levels, keep_original=keep_original)
            # project expanded channels back to original dims expected by embeddings
            self.wproj_enc = nn.Linear(enc_in * mult, enc_in)
            self.wproj_dec = nn.Linear(dec_in * mult, dec_in)
            print(f"[WAVELET/MODEL] on | wavelet={wavelet} L={levels} keep_original={keep_original} mult=x{mult}")
        else:
            print("[WAVELET/MODEL] off")

        # Embeddings (enc_in/dec_in remain ORIGINAL sizes)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention kind
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                               d_model, n_heads, mix=False),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [DecoderLayer(
                AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                               d_model, n_heads, mix=mix),
                AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                               d_model, n_heads, mix=False),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # 1) Wavelet → Project back (ONLY if enabled)
        if self.use_wavelet:
            wx_enc = self.wavetok_enc(x_enc)          # (B, S, enc_in * mult)
            wx_dec = self.wavetok_dec(x_dec)          # (B, S, dec_in * mult)
            # print(wx_enc.shape,wx_dec.shape)
            # Optional one-time debug prints
            if getattr(self, "debug_wavelet", False) and not hasattr(self, "_dbg_printed"):
                print(f"[DBG] wavetok_enc out: {wx_enc.shape} | wavetok_dec out: {wx_dec.shape}")

            x_enc = self.wproj_enc(wx_enc)            # (B, S, enc_in)
            x_dec = self.wproj_dec(wx_dec)            # (B, S, dec_in)

            # Quick guarantees (your asserts)
            assert x_enc.shape[-1] == self.wproj_enc.out_features, f"enc proj mismatch: {x_enc.shape}"
            assert x_dec.shape[-1] == self.wproj_dec.out_features, f"dec proj mismatch: {x_dec.shape}"

            # Optional print after projection (only once)
            if getattr(self, "debug_wavelet", False) and not hasattr(self, "_dbg_printed"):
                print(f"[DBG] wproj_enc out: {x_enc.shape} | wproj_dec out: {x_dec.shape}")
                # If you stored pad info in the tokenizer:
                if hasattr(self.wavetok_enc, "_last_info"):
                    print(f"[DBG] enc pad info: {self.wavetok_enc._last_info}")
                if hasattr(self.wavetok_dec, "_last_info"):
                    print(f"[DBG] dec pad info: {self.wavetok_dec._last_info}")
                self._dbg_printed = True  # prevent spamming every batch

        # 2) From here on, proceed as usual
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)
        dec_out = self.log_softmax(dec_out)

        return (dec_out[:, -self.pred_len:, :], attns) if self.output_attention else dec_out[:, -self.pred_len:, :]




class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention = False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        dec_out = self.log_softmax(dec_out)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
