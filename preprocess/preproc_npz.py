import numpy as np

data = np.load('/Users/ozgun/DataspellProjects/CS401-soffritto/data/H1_features.npz')
print(data.files)
print(len(data.files))

array1 = data['12']
print(array1.shape)
print(array1[0])
