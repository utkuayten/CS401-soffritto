from preprocess.data_loader import RepliSeqFeatureLabelWithCoordinates
from torch.utils.data import DataLoader

chroms = ['22', '9', '17', '7', '13', '20', '8', '15', '19', '18',
          '5', '14', '3', '10', '21', '1', '12', '2', '11', '4', '16', '6']

dataset = RepliSeqFeatureLabelWithCoordinates(
    x_path='data/H1_features.npz',
    y_path='data/H1_labels.npz',
    bedgraph_path='data/H1_coordinates.bedgraph',
    chromosomes=chroms,
    size=(96, 48, 48),
    scale=True
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
    print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
    break
