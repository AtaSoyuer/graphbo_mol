import torch.nn as nn
from graph_env.graph_generator import Graph
from config import *

'''
TRANSFORM FORWARD SO THAT WHEN USING BATCHED GD IN ALGORITMHS.PY,
CAN DIRECTLY ADD THE FEAT_MAT_AGGR_NORMED TO THE DATASET INSIDE BATCH
ST. NO CANT HAVE GRAPH CLASS IN BATCH ERROR
'''

class NN(nn.Module):
    def __init__(self, input_dim, depth, width, aggr_feats = False):
        super(NN, self).__init__()
        self.aggr_feats = aggr_feats
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,width))
        self.layers.append(nn.ReLU())
        for i in range(depth-1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width,1))


    def forward(self, g: Graph):
        if self.aggr_feats:
            feat_mat = torch.from_numpy(g).float().to(device)
        else:
            feat_mat = torch.from_numpy(g).float().to(device)

        x = torch.flatten(feat_mat)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class GNN(nn.Module):
    def __init__(self, input_dim:int, depth:int, width: int, dropout:bool, dropout_prob:float, aggr_feats: bool = True,):
        super(GNN, self).__init__()
        self.aggr_feats = aggr_feats
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,width))
        self.layers.append(nn.ReLU())
        if dropout:
            print("Adding dropout with probability:",dropout_prob)
            self.layers.append(nn.Dropout(dropout_prob))

        #self.scale = torch.nn.Parameter(torch.randn(1))
        #self.scale.requires_grad = True

        for i in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
            if dropout:
                print("Adding dropout with probability:",dropout_prob)
                self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Linear(width,1))

    def forward(self, g: Graph):
        if self.aggr_feats:
            x = g.float().to(device)
        else:
            x = g.float().to(device)

        for i in range(len(self.layers)):
            #print('X_Shape:', x.shape)
            x = self.layers[i](x)
            #print('X_Shape:', x.shape)
        #print('X:', x)
        #return torch.mean(x, dim=0) #* self.scale
        x = torch.mean(x, dim=1)
        #print('X_Shape:', x.shape)
        return torch.mean(x, dim=1)
        #return x.view(x.size(0), -1).mean(1)
    

def normalize_init(net):
    '''
    :param net: input network, random state
    :return: network that is normalized wrt xavier initialization
    '''
    layers_list = [module for module in net.modules() if type(module) == nn.Linear]
    for layer in layers_list[1:]:
        #nn.init.xavier_normal_(layer.weight,gain=np.sqrt(2))
        nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')
        #layer.bias.data.fill_(0.0)
    return net