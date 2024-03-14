import torch.nn as nn
from graph_env.graph_generator import Graph
from config import *

'''
USES ALTERNATIVE GNN MODEL USING PYG LIBRARY AND GCN CONVOLUTIONS AND ALSO UTILIZING THE
BATCH AND EDGE_INDEX ATTRIBUTES OF DATA
'''

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

from torch_geometric.nn import NNConv, Set2Set
from torch.nn import GRU, Linear, ReLU, Sequential

class NNConv_Net(torch.nn.Module):
    def __init__(self, input_dim:int, dim:int, width: int,aggr_feats: bool = False):
        super().__init__()
        self.lin0 = torch.nn.Linear(input_dim, dim)

        nn = Sequential(Linear(5, width), ReLU(), Linear(width, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        #self.alpha = torch.nn.Parameter(torch.tensor(50).to(torch.float32)) #Achieve lazy training by scaling alpha
        #self.alpha.requires_grad = False

        self.alpha = torch.tensor(50, requires_grad=False)

        self.aggr_feats = aggr_feats

    def forward(self, data):

        features = torch.hstack((data.x, data.z[:,None]))

        if self.aggr_feats:
            x = feat_mat_aggr_normed(features,data.edge_index).float().to(device)
        else:
            x = features

        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1) #* self.alpha



# GCN model with 2 layers 
class GNN_pyg(torch.nn.Module):
    def __init__(self, input_dim:int, depth:int, width: int, dropout:bool, dropout_prob:float, aggr_feats: bool = True):
        super(GNN_pyg, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, width))
        #self.layers.append(nn.ReLU())

        for i in range(depth - 1):
            self.layers.append(GCNConv(width, width))

        self.lin = nn.Linear(width,1)

        #print(self.layers)

        self.aggr_feats = aggr_feats    


    def forward(self, data):

        x, edge_index, batch = torch.hstack((data.x,data.pos,data.z[:,None])), data.edge_index, data.batch

        if self.aggr_feats:
            x = feat_mat_aggr_normed(x,edge_index).float().to(device)
        
        for i in range(len(self.layers)):
            #print('X_Shape:', x.shape)
            x = F.relu(self.layers[i](x, edge_index))

        #print(x.shape)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        #print(x.shape)

        #x = F.dropout(x, training=self.training)
        x = self.lin(x)
        #print('X_Shape:',x.shape)
        return x.squeeze()

def normalize_init(net):
    '''
    :param net: input network, random state
    :return: network that is normalized wrt xavier initialization
    '''
    #layers_List = []
    #for module in net.modules():
        #if 
    layers = [layer for layer in net.children()]
    #print(layers)
    base_layers = []
    base_layers.append(layers[0])
    base_layers.append(layers[1].nn[0])
    base_layers.append(layers[1].nn[2])
    base_layers.append(layers[4])
    base_layers.append(layers[5])

    #print(layers[1].nn[0])
    
    #weights = list(layer.parameters())[0]
    #layers_list = [module for module in net.modules() if not isinstance(module, nn.Sequential)]
    #print(layers_list)
    for layer in base_layers:
        #nn.init.xavier_normal_(layer.weight,gain=np.sqrt(2))
        nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')
        #layer.bias.data.fill_(0.0)
    return net

def feat_mat_aggr_normed(features, edge_index): #matrix of \bar h_u
    normed_feats = torch.zeros(features.shape[0], features.shape[1])
    #adj_mat = convert_to_adj(edge_index, features.shape[0])
    for node in range(features.shape[0]):
        #print(edge_index)
        indices = torch.where(edge_index[0,:]==node)[0]
        #print(indices)
        neighbors = edge_index[1,indices]
        #print(neighbors)
        sum_feats = torch.sum(features[neighbors,:], axis=0)
        normed_feats[node,:] = sum_feats / torch.linalg.norm(sum_feats)
        #print('Normed_feats_shape:',normed_feats.shape)
    return normed_feats

#model = NNConv_Net(11,64,8192)
#model = normalize_init(model)

#params = [p for p in model.parameters()]
#print(params)