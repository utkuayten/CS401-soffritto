import numpy as np

# Replace 'results.npy' with the path to your file.
pred = np.load('/Users/utkuayten/Desktop/CS401-soffritto/results/genomic_multitarget_informer/pred.npy')
true = np.load('/Users/utkuayten/Desktop/CS401-soffritto/results/genomic_multitarget_informer/true.npy')

print(pred[0],true[0])

print(pred.shape,true.shape)
