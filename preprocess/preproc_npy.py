import numpy as np

# Replace 'results.npy' with the path to your file.
pred = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/pred.npy')
true = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/true.npy')
metrics = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/results/genomic_multitarget_informer/metrics.npy')
print(pred[0][-1],true[0][-1])

print(pred.shape,true.shape)
print(metrics)