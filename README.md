# DJ GRAPH NETWORK
PyTorch development for Master Thesis

# PyTorch Geometric installation
https://github.com/rusty1s/pytorch_geometric#installation

# Dataset
## Change sequence length of <b>MANIAC</b>.
```
# Creates new dataset with sequence of X
train_set = MANIAC(_FOLDER + "training/", X)
val_set = MANIAC(_FOLDER + "validation/", X)
test_set = MANIAC(_FOLDER + "test/", X)
```

To save the dataset above.
```
# If you want to store multiple/different dataset settings. Just change the name to something suitable for that dataset.
# Eg. This dataset is used for LSTM with a sequence of 4.
with open(os.path.join(_FOLDER + "/raw/lstm_4_training.pt"), 'wb') as f:
            torch.save(train_set, f)
```


Next remove lstm_4_*.pt in MANIAC/processed folder.

DaweSuper(InMemoryDataset) will process MANIAC(Dataset) and create these files (only happens if they doesnt exsist).

To load dataset:
```python
train_dataset = DaweSuper(_FOLDER, "train")
test_ds = DaweSuper(_FOLDER, "test")
valid_ds = DaweSuper(_FOLDER, "valid")
```

To use DataLoader on dataset:
```python
train_loader = DataLoader(train_dataset, batch_size=_bs, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=_bs, shuffle=False)
valid_loader = DataLoader(valid_ds, batch_size=_bs, shuffle=False)
```

NOTE! Additionally to create sequences new list are created (total_list, valid_list, test_list). These list consits of <b>seq_len</b> 
number of graphs.

Example:
```python
# Inside test_list there is X number of sequence arrays
test_list = [ [g1, g2, g3,g g4], [g2, g3, g4, g5], [g3, g4, g5, g6], .... ]

# Randomized with seed(2) and shuffle.
random.shuffle(test_list)

# To visualize the sequence of graphs (req. matplotlib)
show_graphs(test_list, 5)
```


# Add tensorboard logs
Find flag <b>_WRITE</b> and change to True.
```Python
_WRITE = True
```

Find <b>writer</b> too see that it is not hidden.
```python
# Change comment to something useful, eg. 4seq_64conv_lstm_2hidden_with_weights
# So we can easily access and see difference in tensorboard
writer = SummaryWriter(comment="seq_64conv_dropout0.5_no_weights")
```


# Model
All model is created as defined in the PyTorch Geometrics. 

Good example: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs

## Current model

```Python
dim = 32
hidden_size = 1
input_size = 4
batch_s = 1
output_size = 8

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.lin0 = torch.nn.Linear(22, dim)
        
        nn = Sequential(Linear(2, 37), ReLU(), Linear(37, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, output_size)
        self.lin1 = torch.nn.Linear(dim, 8)
        self.prev_hidden = None
        
    def forward(self, data):
        prev_hidden = None
        seq_l = len(data)
        g_list = []
        for d in data:
            d = d.to(device)
            x, edge_index, edge_attr, batch = d.x, d.edge_index, d.edge_attr, d.batch
            batch_size = batch.max().item() + 1

            # Conv
            out = F.elu(self.lin0(x))
            out = F.elu(self.conv(out, edge_index, edge_attr))

            x_1 = scatter_mean(out, batch, dim=0)
            out = x_1[0]

            g_list.append(out)

        if self.prev_hidden is None:
            self.prev_hidden = (torch.zeros(output_size, batch_size, hidden_size).cuda(),
                                torch.zeros(output_size, batch_size, hidden_size).cuda())

        my_tensor = pad_sequence(g_list)
        my_tensor = my_tensor.to(device)
        self.prev_hidden = repackage_hidden(self.prev_hidden)
        q, h = self.lstm( my_tensor.view(len(my_tensor), 1, -1), self.prev_hidden)
        self.prev_hidden = h
        
        out = self.lin1(out)
        
        return out
```

