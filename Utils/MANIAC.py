from torch.utils.data import Dataset

'''

    Creates MANIAC dataset to work with PyTorch Geometric.

'''
class MANIAC(Dataset):
    def __init__(self, root_dir, window, temporal=False):
        self.window = window
        self.root_dir = root_dir
        self.all_xmls = find_xmls(self.root_dir)
        self.sp = SECParser()
        self.temporal = temporal

        for xml in self.all_xmls:
            self.dict_with_graphs = self.sp(xml)

        self.samples = create_big_list(self.dict_with_graphs)

    def __len__(self):
        return len(self.samples) - self.window

    def __getitem__(self, idx):

        if self.window > 0:

            x = self.samples[idx:idx+self.window]

            current_action = self.samples[idx].graph['features']
            step_back = 0

            for i in range(self.window):
                if self.samples[idx+i].graph['features'] != current_action:
                    step_back += 1

            if step_back > 0:
                x = self.samples[idx-step_back:idx+self.window-step_back]
            else:
                x = self.samples[idx:idx+self.window]
        else:
            x = self.samples[idx]

        if self.temporal:
            return util.concatenateTemporal(x, _relations, spatial_map)
        else:
            return x

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)

'''

    Creates MANIAC InMemory dataset to work with PyTorch Geometric.

    Read more about it at
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets

'''
class ManiacIMDS(InMemoryDataset):
    def __init__(self,
                root,
                dset="train",
                transform=None):
        super(ManiacDS, self).__init__(root, transform)

        if dset == "train":
            path = self.processed_paths[0]
        elif dset == "valid":
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['training_' + str(_TIME_WINDOW) + 'w.pt', 'validation_' + str(_TIME_WINDOW) + 'w.pt', 'test_' + str(_TIME_WINDOW) + 'w.pt']

    @property
    def processed_file_names(self):
        return ['training_' + str(_TIME_WINDOW) + 'w.pt', 'validation_' + str(_TIME_WINDOW) + 'w.pt', 'test_' + str(_TIME_WINDOW) + 'w.pt']

    def download(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def process(self):
        big_slices = []
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            big_data = []
            graphs = torch.load(raw_path)

            # Creates torch_geometric data from networkx graphs
            # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
            for graph in graphs:
                G = nx.convert_node_labels_to_integers(graph)
                G = G.to_directed() if not nx.is_directed(G) else G
                edge_index = torch.tensor(list(G.edges)).t().contiguous()

                data = {}

                for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
                    for key, value in feat_dict.items():
                        data[key] = [value] if i == 0 else data[key] + [value]

                for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
                    for key, value in feat_dict.items():
                        data[key] = [value] if i == 0 else data[key] + [value]

                for key, item in data.items():
                    try:
                        data[key] = torch.tensor(item)
                    except ValueError:
                        pass

                # Creates the tg data
                data['edge_index'] = edge_index.view(2, -1)
                data = tg.data.Data.from_dict(data)
                data.y = torch.tensor(graph.graph['features'])

                # This is not used, can be useful in future development if the sequence id is needed.
                #data.seq = torch.tensor([graph.graph['seq']])

                if _SKIP_CONNECTIONS:
                    if data.edge_attr is not None:
                        big_data.append(data)
                else:
                    big_data.append(data)

            for graph in big_data:
                if graph.edge_attr is None:
                    print(graph)
                    print(graph.edge_attr)
                    print(action_map[graph.y.argmax().item()])
                    break

            torch.save(self.collate(big_data), path)
