import numpy as np

data = np.load('data/H1_labels.npz')
print(data.files)
print(len(data.files))

array1 = data['12']
print(array1.shape)
print(array1[0])
print(array1[1])
print(array1[2])
print(array1[3])
