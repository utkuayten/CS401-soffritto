import numpy as np

from train_intra_cell_line import GAT_intracell

train_chroms = [str(i) for i in range(1, 23) if str(i) != "9"]
test_chrom = "9"

trainer = GAT_intracell(
    features_file="GAT/data/H1_features.npz",
    labels_file="GAT/data/H1_labels.npz",
    train_chromosomes=train_chroms,
    test_chromosome=test_chrom,

    # must match your updated utils signature
    hop_list=(1,2,4,6,8,16),
    chroms_per_epoch=2,
    epochs= 500,
    hidden_dim = 128,
    heads = 9,
    layers = 2,
    dropout = 0.10,
)

trainer.fit()

pred = trainer.predict_test_probs()
np.save("H1_chr9_pred_intra_cell_line_GAT.npy", pred)
print("saved:", pred.shape)