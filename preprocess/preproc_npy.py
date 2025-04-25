import numpy as np

# Replace 'results.npy' with the path to your file.
pred = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/pred.npy')
true = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/true.npy')
#pred2 = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/data/H1_chr9_pred_intra_cell_line.npy')
metrics = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/metrics.npy')
print(pred)
print(true)
print(pred.shape)
print(true.shape)
print(metrics)