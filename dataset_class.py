import torch

class QM9_GNNUCB_Dataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):

        self.graph_features = []
        self.rewards = []
        self.transform = transform

    def __len__(self):
        return len(self.graph_features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        graph = self.graph_features[idx]
        reward = self.rewards[idx]

        sample = {'graph': graph, 'reward': reward}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def add(self, graph, reward): #GNNUCB selects and adds only one pair  in each step

        self.graph_features.append(graph)
        self.rewards.append(reward)


# graphs = [torch.tensor([1,1,1]), torch.tensor([2,2,2]), torch.tensor([3,3,3]), torch.tensor([4,4,4])]
# rewards = [torch.tensor(7), torch.tensor(8), torch.tensor(9), torch.tensor(10)]

# QM9_Dataset = QM9_GNNUCB_Dataset()

# for i in range(len(graphs)):
#     QM9_Dataset.add(graphs[i], rewards[i])

# print(QM9_Dataset.__len__())
# print(QM9_Dataset[0])
# print(QM9_Dataset[1])
# print(QM9_Dataset[2])
# print(QM9_Dataset[3])