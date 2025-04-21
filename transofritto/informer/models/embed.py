import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class GenomicEmbedding(nn.Module):
    def __init__(self, d_model, num_chroms=25):
        super(GenomicEmbedding, self).__init__()

        self.chrom_embed = FixedEmbedding(c_in=num_chroms, d_model=d_model)  # e.g., chroms 1–22, X=23, Y=24
        self.pos_proj = nn.Linear(1, d_model)  # for start coordinate

        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x_mark):
        """
        x_mark: [B, L, 2] → [chrom, start]
        """
        chrom = x_mark[:, :, 0].long()            # categorical [B, L]
        pos = x_mark[:, :, 1:2].float()           # continuous [B, L, 1]

        # Normalize position (optional but recommended)
        pos = pos / 3e9  # genome-wide normalization

        chrom_embed = self.chrom_embed(chrom)     # [B, L, d_model]
        pos_embed = self.pos_proj(pos)            # [B, L, d_model]

        combined = torch.cat([chrom_embed, pos_embed], dim=-1)  # [B, L, 2*d_model]
        return self.proj(combined)  # [B, L, d_model]

NUM_COORD_BINS = 100   # number of genome coordinate bins
NUM_CHR_BINS   = 24   # number of chromosome bins

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # Choose fixed sinusoidal or learned lookup
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # Embeddings for genomic features
        self.coord_embed = Embed(NUM_COORD_BINS, d_model)
        self.chr_embed   = Embed(NUM_CHR_BINS,   d_model)

    def forward(self, x):
        """
        x: FloatTensor of shape [B, L, 2] where
           x[...,0] = normalized chromosome number ∈ [-0.5, 0.5]
           x[...,1] = normalized genomic coordinate    ∈ [-0.5, 0.5]
        """
        # Split continuous features
        chr_norm   = x[..., 0]
        coord_norm = x[..., 1]

        # Discretize into integer indices
        chr_idx = ((chr_norm + 0.5) * (NUM_CHR_BINS - 1)) \
            .round().long().clamp(0, NUM_CHR_BINS - 1)
        coord_idx = ((coord_norm + 0.5) * (NUM_COORD_BINS - 1)) \
            .round().long().clamp(0, NUM_COORD_BINS - 1)
        # Lookup embeddings and sum
        chr_x   = self.chr_embed(chr_idx)
        coord_x = self.coord_embed(coord_idx)

        return chr_x + coord_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3, 'g':2}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        #print(x)
        return self.embed(x)

class GenomicEmbedding(nn.Module):
    """
    Embeds a 3‑column genomic coordinate tensor [chrom, start, end]
    into a d_model vector by summing:
      - a learned Embedding(chrom)
      - a sinusoidal Fourier embed of start
      - a sinusoidal Fourier embed of end
    """
    def __init__(self, num_chroms: int, d_model: int):
        super().__init__()
        # 1) learned categorical for chromosome
        self.chrom_emb = nn.Embedding(num_chroms, d_model)

        # 2) precompute the inv_freq for half of d_model
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)  # (d_model/2,)

    def forward(self, x):
        # x: (B, L, 3)  where x[:,:,0]=chrom, x[:,:,1]=start, x[:,:,2]=end
        chrom = x[:, :, 0].long()   # (B,L)
        start = x[:, :, 1].float()  # (B,L)
        end   = x[:, :, 2].float()  # (B,L)

        # Embed chrom
        emb_chrom = self.chrom_emb(chrom)  # (B,L,d_model)

        # Fourier features for start
        #  start.unsqueeze(-1): (B,L,1) * inv_freq (d_model/2) → (B,L,d_model/2)
        freqs = start.unsqueeze(-1) * self.inv_freq
        emb_start = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # (B,L,d_model)

        # Fourier features for end
        freqs2 = end.unsqueeze(-1) * self.inv_freq
        emb_end = torch.cat([torch.sin(freqs2), torch.cos(freqs2)], dim=-1)   # (B,L,d_model)

        return emb_chrom + emb_start + emb_end   # (B,L,d_model)
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='w', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # add a special branch:
        if embed_type == 'geno' or embed_type == 'timeF':
            # simple linear: 3 → d_model
            self.temporal_embedding = TimeFeatureEmbedding(d_model, embed_type, freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        #print(self.value_embedding,self.position_embedding,self.temporal_embedding)
        return self.dropout(x)