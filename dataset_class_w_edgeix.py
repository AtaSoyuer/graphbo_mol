import torch
import torch_geometric
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader
from torch_geometric.datasets import QM9
import os.path as osp
#import numpy as np


import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops


class QM9_GNNUCB_Dataset(torch_geometric.data.InMemoryDataset):

    def __init__(self, transform=None):

        self.graph_features = []
        self.rewards = []
        self.edge_indices = []
        self.num_nodes = []
        self.transform = transform

    def __len__(self):
        return len(self.graph_features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        graph = self.graph_features[idx]
        reward = self.rewards[idx]
        edge_index = self.edge_indices[idx]
        num_nodes = self.num_nodes[idx] #TO KEEP TRACK OF NUM NODES IN EACH GRAPH ADDED SO THAT WE CAN CREATE THE BATCH PARAMETER INLOADER

        sample = {'graph': graph, 'reward': reward, 'edge_index': edge_index, 'num_nodes': num_nodes}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def add(self, graph, reward, edge_index): #GNNUCB selects and adds only one pair  in each step

        self.graph_features.append(graph)
        self.rewards.append(reward)
        self.edge_indices.append(edge_index)
        self.num_nodes.append(graph.shape[0])

class MyTransform(object):
    def __init__(self, target):
        self.target = target
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data
    
# class LaplacianFeatures(object):
#     def __init__(self, features):
#         self.laplacian = torch_geometric.trans
#     def __call__(self, data)

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
    
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# transform = T.Compose([MyTransform(0), Complete(), T.Distance(norm=False), T.AddLaplacianEigenvectorPE(k=3, attr_name=None, is_undirected=True)])
# dataset = QM9(path, transform=transform)

# print(dataset[0])

# class QM9_Dataset(torch_geometric.data.InMemoryDataset):

#     def __init__(self, target):

#         self.path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
#         self.transform = T.Compose([MyTransform(target), Complete(), T.Distance(norm=False)])
#         self.dataset = QM9(self.path, transform=self.transform)

#     def __len__(self):

#         return len(self.dataset)

#     def __getitem__(self, idx):

#         return self.dataset[idx]

# indices = np.array([1,3,5])
# target = 0
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# transform = T.Compose([MyTransform(target), Complete(), T.Distance(norm=False)])

# dataset = QM9(path, transform=transform)

# dataset2 = QM9(path, transform=transform)

# dataset_subset = dataset[indices]
# dataset_subset_2 = dataset2[indices]

# print(dataset_subset[0], dataset_subset[1], dataset_subset[2])

# print(dataset_subset_2[0], dataset_subset_2[1], dataset_subset_2[2])

# print(dataset[0])

# dataloader = DataLoader(dataset, batch_size=25, shuffle=True)

# for i, data in enumerate(dataloader):
#     pass

# print(dataset[0])
#a = QM9_Dataset(0)

#print(a.dataset.data.y)

#print(a[0,2,55][2])

#print(a.__len__())

#print(a[0])
   
# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length

#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     ## get sequence lengths
#     #lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
#     ## padd
#     #batch = zip(*batch)
#     #print(len(batch))
#     data = torch.vstack([ torch.Tensor(t['graph']) for t in batch ]).float()
#     #data = torch.nn.utils.rnn.pad_sequence(data).permute((1,0,2)) #PERMUTE BECAUSE THEIS PADDING FUNCTION
#     #PUTS BATCH DIMENSION IN 2ND DIM FOR SOME REASON!!!
#     #print('Collate data shape:', data.shape)
#     #data = torch.cat(data, dim=0)
#     rewards = torch.hstack([ torch.Tensor(t['reward']) for t in batch ]).float()
#     edge_indices = torch.hstack([ t['edge_index'].type(torch.int64) for t in batch ]).type(torch.int64)

#     batch_var = torch.hstack([torch.ones(t['num_nodes'])*i for i, t in enumerate(batch) ]).type(torch.int64)
#     ## compute mask
#     ##mask = (batch != 0).to(device)
#     #print("yes")
#     return data, rewards, edge_indices, batch_var

#     batch_var = torch.hstack([torch.ones(t['num_nodes'])*i for i, t in enumerate(batch) ])
#     ## compute mask
#     ##mask = (batch != 0).to(device)
#     #print("yes")
#     return data, rewards, edge_indices, batch_var

# graphs = [torch.tensor([[1,1,1]]), torch.tensor([[2,2,2],[2,2,2]]), torch.tensor([[3,3,3],[3,3,3]]), torch.tensor([[4,4,4],[4,4,4],[4,4,4],[4,4,4]])]
# rewards = [torch.tensor(7), torch.tensor(8), torch.tensor(9), torch.tensor(10)]
# edge_indices = [torch.tensor([[1],[1]]), torch.tensor([[2,2,2],[2.5,2.5,2.5]]), torch.tensor([[3,3,3],[3.5,3.5,3.5]]), torch.tensor([[4,4,4],[4.5,4.5,4.5]])]

# QM9_Dataset = QM9_GNNUCB_Dataset()

# for i in range(len(graphs)):
#     QM9_Dataset.add(graphs[i], rewards[i], edge_indices[i])

# #print(QM9_Dataset.__len__())
# #print(QM9_Dataset[0])
# #print(QM9_Dataset[1])
# #print(QM9_Dataset[2])
# #print(QM9_Dataset[3])

# loader = DataLoader(QM9_Dataset, batch_size=2, shuffle=False, collate_fn = collate_fn_padd)

# for i, (data, rewards, edge_indices, batch) in enumerate(loader):
#     print('Data:', data)
#     print('Rewards:', rewards)
#     print('Edge_indices:', edge_indices)
#     #print('Edge_indices_shape:', edge_indices.shape)
#     print("Batch:", batch)

# print(list(enumerate(loader))[0])