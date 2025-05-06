# Transofritto

**Transofritto** is a Transformer-based model for predicting high-resolution DNA replication timing (16-fraction Repli-Seq profiles).  
It is inspired by [Soffritto](https://github.com/ay-lab/Soffritto) [1], which uses LSTMs to model genomic signals.  
Transofritto builds on this idea using the **Informer** architecture [2] to better capture long-range dependencies.

> 📖 References  
> [1] D. Bolzan, F. Ay, *Soffritto: a deep-learning model for predicting high-resolution replication timing*, bioRxiv, 2025. [https://doi.org/10.1101/2025.01.23.634644](https://doi.org/10.1101/2025.01.23.634644)  
> [2] Haoyi Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. [https://arxiv.org/abs/2012.07436](https://arxiv.org/abs/2012.07436)

---

## Overview

Transofritto predicts the distribution of replication timing across 16 S-phase fractions for each 50kb bin using the following inputs:

- 2-fraction Repli-Seq (log2(Early/Late))
- 6 Histone modification signals (H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3)
- GC content
- Gene density

---

## Data Sources

- 16-Fraction Repli-Seq: [GSE137764](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137764)
- Histone ChIP-seq: [ENCODE Project](https://www.encodeproject.org)
- 2-Fraction RT: [4DN Data Portal](https://data.4dnucleome.org/)
- Gene Annotations: [GENCODE](https://www.gencodegenes.org/)

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run transofritto intracell

```bash
python3 transofritto/run_model.py \
  --model informer \
  --setting multitarget \
  --root_path transofritto/data \
  --data_path H1_genomic.csv \
  --target target_1 \
  --seq_len 32 \
  --label_len 16 \
  --pred_len 1 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_chroms 1 2 3 4 5 6 7 8 9 10 11 12 \
  --val_chroms 13 14 \
  --checkpoints ./checkpoints \
  --weight_decay 0.001
```
