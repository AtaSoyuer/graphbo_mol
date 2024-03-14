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
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn import NNConv, Set2Set
from torch.nn import GRU, Linear, ReLU, Sequential

# class NNConv_Net(torch.nn.Module):
#     def __init__(self, input_dim:int, dim:int, width: int,):
#         super().__init__()
#         self.lin0 = torch.nn.Linear(input_dim, dim)

#         nn = Sequential(Linear(5, width), ReLU(), Linear(width, dim * dim))
#         self.conv = NNConv(dim, dim, nn, aggr='mean')
#         self.gru = GRU(dim, dim)

#         self.set2set = Set2Set(dim, processing_steps=3)
#         self.lin1 = torch.nn.Linear(2 * dim, dim)
#         self.lin2 = torch.nn.Linear(dim, 1)

#         #self.alpha = torch.nn.Parameter(torch.tensor(50).to(torch.float32)) #Achieve lazy training by scaling alpha
#         #self.alpha.requires_grad = False

#         self.alpha = torch.tensor(50, requires_grad=False)

#     def forward(self, data):
#         out = F.relu(self.lin0(data.x))
#         h = out.unsqueeze(0)

#         for i in range(3):
#             m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
#             out, h = self.gru(m.unsqueeze(0), h)
#             out = out.squeeze(0)

#         out = self.set2set(out, data.batch)
#         out = F.relu(self.lin1(out))
#         out = self.lin2(out)
#         return out.view(-1) * self.alpha



# GCN model with 2 layers 
class GNN_pyg(torch.nn.Module):
    def __init__(self, input_dim:int, depth:int, width: int, dropout:bool, dropout_prob:float, aggr_feats: bool = True):
        super(GNN_pyg, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, width))
        #self.layers.append(nn.ReLU())

        for i in range(depth - 1):
            self.layers.append(GCNConv(width, width))
            if dropout:
                print("Adding dropout with probability:",dropout_prob)
                self.layers.append(nn.Dropout(dropout_prob))

        self.lin = nn.Linear(width,1)
        self.layers.append(self.lin)


    def forward(self, data):

        x, edge_index, batch = torch.hstack((data.x,data.pos,data.z[:,None])), data.edge_index, data.batch

        for i in range(len(self.layers)-1):
            #print('X_Shape:', x.shape)
            x = F.relu(self.layers[i](x, edge_index))

        #print(x.shape)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        #x = global_add_pool(x, batch)
        #print(x.shape)

        #x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return x
    
class GNN_old(nn.Module):
    def __init__(self, input_dim:int, depth:int, width: int, dropout:bool, dropout_prob:float, batch_size: int, aggr_feats: bool = True,):
        super(GNN_old, self).__init__()
        self.batch_size = batch_size
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

    def forward(self, g):

        features = torch.hstack((g.x, g.pos, g.z[:,None]))
        if self.aggr_feats:
            x = feat_mat_aggr_normed(features, g.edge_index).float().to(device)
        else:
            x = feat_mat_aggr_normed(features, g.edge_index).float().to(device)

        #if split:
            #x = torch.split(x, self.batch_size)

        for i in range(len(self.layers)):
            #print('X_Shape:', x.shape)
            x = self.layers[i](x)
            #print('X_Shape:', x.shape)
        #print('X_shape:', x.shape)
        return torch.mean(x, dim=0) #* self.scale
        return torch.sum(x, dim=0) #* self.scale
        #x = global_mean_pool(x, g.batch)
        #x = global_add_pool(x, g.batch)
        #x = torch.mean(x, dim=0)
        #print('X_Shape:', x.shape)

        #if split:
            #x = torch.mean(x, dim=1)
        #return x.squeeze() #batch_sizex1 ----> batch_size
        #return x.view(x.size(0), -1).mean(1)

# def convert_to_adj(pairs, num_nodes):
#     pairs = torch.transpose(pairs,0,1)
#     zero = torch.zeros(num_nodes, num_nodes)
#     zero[pairs[:,0], pairs[:,1]] = 1 
    
#     return zero

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

def normalize_init(net):
    '''
    :param net: input network, random state
    :return: network that is normalized wrt xavier initialization
    '''
    layers_list = []
    for module in net.modules():
        #print(module)
        if type(module) == nn.Linear:
            #layers_list.append(module)
            nn.init.kaiming_normal_(module.weight, mode = 'fan_in', nonlinearity = 'relu')
            if module.bias is not None:
                #print('linbias')
                nn.init.normal_(module.bias, std=0.1)
        if type(module) == GCNConv:
            #layers_list.append(module.lin)
            nn.init.kaiming_normal_(module.lin.weight, mode = 'fan_in', nonlinearity = 'relu')
            if module.bias is not None:
                #print('gcnbias')
                nn.init.normal_(module.bias, std=0.1)
    #layers_list = [module if type(module) == nn.Linear else module.lin if type(module) == GCNConv else pass for  module in net.modules()]
    #print('Layers list:',layers_list)
    #for layer in layers_list[1:]:
        #nn.init.xavier_normal_(layer.weight,gain=np.sqrt(2))
        #nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'relu')
        #layer.bias.data.fill_(0.0)
    return net

#model = GNN_pyg(15,2,256,aggr_feats=False,dropout=False,dropout_prob=0.2)
#layers_list = [module for module in model.modules() if type(module) == nn.Linear or type(module) == GCNConv]
#print(layers_list[1].lin)

#model = normalize_init(model)


#params = [p for p in model.parameters()]
#print(params)