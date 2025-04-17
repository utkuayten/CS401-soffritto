import numpy as np

# Replace 'results.npy' with the path to your file.
pred = np.load('/Users/ozgun/DataspellProjects/Soffritto/predictions/H1_chr9_pred_intra_cell_line.npy')
true = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/true.npy')
pred2 = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/data/H1_chr9_pred_intra_cell_line.npy')
metrics = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/metrics.npy')
print(pred2)

print(pred2.shape)