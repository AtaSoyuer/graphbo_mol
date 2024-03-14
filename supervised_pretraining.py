import os.path as osp
print(osp.abspath("."))

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from matplotlib import pyplot as plt

import os

import copy
import math

import numpy as np
import argparse
import torch.nn as nn

import gpytorch
from nets_temp import GNN_pyg as GNTK_model
from nets_temp import normalize_init
from torch_geometric.loader import DataListLoader
from sklearn.preprocessing import Normalizer

from dataset_class_w_edgeix import MyTransform, Complete 
import os.path as osp
import sys
sys.path.append(os.path.abspath("./plot_scripts/")) 
import bundles
import torch_geometric.transforms as T

MAX_NUM_NODES = 29

def feat_pad(feat_mat):
    return torch.nn.functional.pad(feat_mat,pad=(0,0,0,MAX_NUM_NODES-len(feat_mat)), value=0)#value=float('nan'))
    
def z_pad(feat_mat):
    return torch.nn.functional.pad(feat_mat,pad=(0,MAX_NUM_NODES-len(feat_mat)), value=0)# value=float('nan'))   


class Net_NNCONV(torch.nn.Module):
    def __init__(self, input_dim:int, width:int, dim:int):
        super().__init__()
        self.lin0 = torch.nn.Linear(input_dim, dim)

        nn = Sequential(Linear(5, width), ReLU(), Linear(width, dim * dim)) #NN learns dimxdim matrices to map edge_features to s.t. these learnt matrices are used to multiply hidden states in message passing
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=5)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        #self.alpha = torch.nn.Parameter(torch.tensor(50).to(torch.float32)) #Achieve lazy training by scaling alpha
        #self.alpha.requires_grad = False

    def forward(self, data):
        out = F.relu(self.lin0(torch.hstack((data.x, data.z[:,None])))) #Just some initial layer to precess raw feature matrix
        h = out.unsqueeze(0) #Basically same as out, just added 1 more dimenstion s.t. h = [[out]] s.t. it can now be inputted to GRU

        for i in range(5): #T in paper, num_procesing steps
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr)) #Message passing phase by the NNCONV, out=h^t-1, m = m^t
            out, h = self.gru(m.unsqueeze(0), h) #Eqn at left bottom of page 3 in paper, this is the vertex update function using a Gate Recurrent Unit with update, reset gates etc. to produce the updated hidden state of that vertex h, here, all hiden states of vertices stacked ar thought as sequences
            out = out.squeeze(0) #Readjust dimensions as it will be inputted to NNConv again

        out = self.set2set(out, data.batch) #Use to add more expressive power in addition to just passing final hidden
        #states thru a linear layer. Hidden states generated are decoded to map each sequence to a sequence of possibly different length.
        #in this case in particulat, sequences of batch_size to obtain a final hidden state for each graph based on its individual vertex hidden states.
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1) #* self.alpha

class GNTK(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    def __init__(self, GNTK_model, device = None, **kwargs):
        super().__init__(**kwargs)

        self.model = normalize_init(GNTK_model)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # @property
        # def device(self):
        #     return self.device

    # this is the kernel function
    def forward(self, graph_1, graph_2, **params):
        graph_pairs = [graph_1, graph_2]
        NTK_features = []
        for G in graph_pairs:
            self.model.zero_grad()
            self.model(G.cuda()).backward(retain_graph=True) #Compute gradients wrt the last forward pass,
            #since we do multiple backwards on the same computational graph (with gradients cleared at each step) iteratively,
            #so we dont want the implicit computattions in the networw pretaining to f0 to be freed!!!
            # Get the Variance.
            NTK_features.append(torch.cat([p.grad.flatten().detach() for p in self.model.parameters()]).cpu())
            #Flatten the gradients wrt each parameter anc concatenates them end to end to gett a full gradient vecotr, g_theta(Graph)
            return torch.inner(NTK_features[0], NTK_features[1]) / 256.0
    
class GP_gpy(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train,likelihood):
        super(GP_gpy, self).__init__(x_train, y_train, likelihood)
        #self.GNTK_model = GNTK_model
        self.mean = gpytorch.means.ConstantMean() # Construct the mean function
        #self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        #self.cov = GNTK(GNTK_model=self.GNTK_model) # Construct the kernel function
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

class GNTK_features():

    def __init__(self, GNTK_model):

        self.model = normalize_init(GNTK_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # this is the kernel function
    def compute(self, data):
        NTK_features = []
        for G in data:
            self.model.zero_grad()
            self.model(G.to(self.device)).backward(retain_graph=True) #Compute gradients wrt the last forward pass,
            #since we do multiple backwards on the same computational graph (with gradients cleared at each step) iteratively,
            #so we dont want the implicit computattions in the networw pretaining to f0 to be freed!!!
            # Get the Variance.
            NTK_features.append(torch.cat([p.grad.flatten().detach() for p in self.model.parameters()]).cpu())
            #Flatten the gradients wrt each parameter anc concatenates them end to end to gett a full gradient vecotr, g_theta(Graph)
        return torch.stack(NTK_features)

class GP():

    def __init__(self, sigma, mean, kernel_features):

        self.dtype = torch.float64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mean = torch.tensor(mean).to(self.device) # Construct the mean function
        self.sigma = torch.tensor(sigma).to(self.device)
        self.inv_mat = None

        self.kernel_features = kernel_features

        self.G_mat = None
        self.K_mat = None
        self.K_inv = None

        self.data = {
            'graph_indices': [], #GRAPH DATA
            'rewards': [], #CONTINOUS REWARD
            'weights': [],
            'means': [] #PREDICTED MEANS FOR EACH SAMPLE IN THE DATASET, UPDATED ONCE EVERY BATCH_SIZE STEPS!!
        }
        
    def fit(self,data):

        self.data['graph_indices'].extend(data)
        
        g_to_add = self.kernel_features.compute(data).to(self.device).to(dtype=self.dtype)

        if self.G_mat is None:
            self.G_mat = (g_to_add.reshape(len(data), -1).to(self.device) / np.sqrt(256)).to(dtype=self.dtype)
        else:
            self.G_mat = torch.cat((self.G_mat, g_to_add.to(self.device) / np.sqrt(256)), dim=0).to(dtype=self.dtype)
        
        self.K_mat = torch.matmul(self.G_mat, self.G_mat.t()).to(dtype=self.dtype).to(self.device)
        
        self.K_inv = (torch.inverse(torch.diag(torch.ones(self.G_mat.shape[0]).to(self.device) * (self.sigma)**2) \
                                            + self.K_mat).to(dtype=self.dtype)).to(self.device)
        
    def predict(self,data):

        print(len(self.data['graph_indices']))

        g_vectors = self.kernel_features.compute(data).to(self.device).to(dtype=self.dtype)

        g_vectors_observed = self.kernel_features.compute(self.data['graph_indices']).to(self.device).to(dtype=self.dtype)

        if len(self.data['graph_indices']) <= 1:
            kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=self.dtype) / 256)
            print('Kxt shape:', kx_t_matrix.shape)
            post_vars = torch.sqrt(torch.sum( g_vectors * g_vectors_observed, dim=1 ) / 256 - torch.sum(kx_t_matrix * self.K_inv.to(self.device) * kx_t_matrix, dim=1))
        else:
            kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=self.dtype) / 256)
            print('Kxt shape:', kx_t_matrix.shape)
            post_vars = torch.sqrt(torch.sum( g_vectors * g_vectors, dim=1 ) / 256 - torch.sum(torch.matmul(kx_t_matrix, self.K_inv.to(self.device)) * kx_t_matrix, dim=1))

        y_vector = []
        mu_a = (torch.ones(len(self.data['graph_indices'])).to(dtype=self.dtype).to(self.device)*self.mean).to(dtype=self.dtype).to(self.device)
        mu_prev = (torch.ones(len(data)).to(dtype=self.dtype).to(self.device)*self.mean).to(dtype=self.dtype).to(self.device)

        for G in self.data['graph_indices']:
            y_vector.append(G.y.cpu().item())

        y_vector = torch.tensor(y_vector).to(dtype=self.dtype).to(self.device)

        #print('y_vec:', y_vector.shape)
        #print('mu_a:', mu_a.shape)

        if len(self.data['graph_indices']) <= 1:
            post_means = mu_prev + torch.sum(kx_t_matrix * self.K_inv.to(self.device) * (y_vector - mu_a), dim=1)
        else:
            post_means = mu_prev + torch.sum(torch.matmul(kx_t_matrix, self.K_inv.to(self.device)) * (y_vector - mu_a), dim=1)

        return post_means, post_vars




class SupervisedPretrain:  #THE DEFEAULT PRETRAINING ROUTINE FOR GNN-UCB

    def __init__(self, target:int=0, dim:int=64, input_dim:int=12, width:int=128, reward_plot_dir:str='reward_0', pretrain_indices_name:str='pretrain_indices_rew0', \
                 model_name:str='nnconv_reward0_8000samples_100ep', num_indices:int=7000, pretraining_load_pretrained=False, \
                 pretraining_pretrain_model_name='nnconv_reward4_8000samples_100ep', laplacian_features=False, laplacian_k=1, std = None, dataset=None):
        
        self.target = target
        self.dim = dim
        self.width = width
        self.reward_plot_dir = reward_plot_dir
        self.pretrain_indices_name = pretrain_indices_name
        self.model_name = model_name
        self.pretraining_load_pretrained = pretraining_load_pretrained
        self.pretraining_pretrain_model_name = pretraining_pretrain_model_name
        self.laplacian_features = laplacian_features
        self.laplacian_k = laplacian_k
        self.num_indices = num_indices
        self.std = std
        self.dataset = dataset


        if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/'):
            pass
        else:
            os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/')

        self.input_dim = input_dim

        # Split datasets.
        self.subset_indices = np.random.choice(len(self.dataset), self.num_indices, replace=False)
        dataset_subset = dataset[self.subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES

        self.test_dataset = dataset_subset[:int(self.num_indices/10)]
        self.val_dataset = dataset_subset[int(self.num_indices/10):int(self.num_indices/7)]
        self.train_dataset = dataset_subset[int(self.num_indices/7):]
        self.test_loader = DataLoader(self.test_dataset, batch_size=25, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=25, shuffle=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=25, shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net_NNCONV(input_dim = self.input_dim, width = self.width, dim = self.dim).to(self.device)

        if self.pretraining_load_pretrained:
            #self.func = torch.load('/local/bsoyuer/base_code/graph_BO/results/saved_models/reward1_5epochs')
            #self.func.load_state_dict(torch.load('/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/nnconv_reward0_1000samples_100ep.pt'))
            self.model.load_state_dict(torch.load(f'/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.pretraining_pretrain_model_name}.pt'))
            self.model.train()
            print(f"Loaded Pretrained Model From /cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.pretraining_pretrain_model_name}.pt")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                            factor=0.7, patience=5,
                                                            min_lr=0.00001)

    def train(self, epoch):
        self.model.train()
        loss_all = 0
        ct = 0

        for data in self.train_loader:
            #print(ct)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(data), data.y)
            #loss = F.mse_loss(model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            self.optimizer.step()
            ct += 1
        print('Count:',ct)
        return loss_all / len(self.train_loader.dataset)


    def test(self, loader):
        self.model.eval()
        error = 0

        for data in loader:
            data = data.to(self.device)
            error += (self.model(data) * self.std - data.y * self.std).abs().sum().item()  # MAE
            #error += (model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch) * std - data.y * std).abs().sum().item()
        return error / len(loader.dataset)
    
    def train_loop(self):

        best_val_error = None
        for epoch in range(1, 101):
            #print(model.alpha)
            lr = self.scheduler.optimizer.param_groups[0]['lr']
            loss = self.train(epoch)
            val_error = self.test(self.val_loader)
            self.scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                test_error = self.test(self.test_loader)
                best_val_error = val_error

            print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
                f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')
            
    def print_and_plot(self):
    
        state_dict = self.model.state_dict()
        torch.save(state_dict, f'/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.model_name}.pt')
        np.save(f'/cluster/scratch/bsoyuer/base_code/graph_BO/{self.pretrain_indices_name}.npy', self.subset_indices)

        plot_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)

        means = []
        graph_rewards = []

        with torch.no_grad():
            for i, data in enumerate(plot_loader):
                outputs = self.model(data.to(self.device))
                #outputs = model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch)
                rewards = data.y

                means.append(outputs)
                graph_rewards.append(rewards)

        max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards).flatten()), torch.max(torch.tensor(means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards).flatten()), torch.min(torch.tensor(means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), c=torch.tensor(graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Training set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_example.jpg')

        plot_loader_val = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        val_means = []
        val_graph_rewards = []

        with torch.no_grad():
            for i, data in enumerate(plot_loader_val):
                outputs = self.model(data.to(self.device))
                #outputs = model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch)
                rewards = data.y

                val_means.append(outputs)
                val_graph_rewards.append(rewards)

        max_val = torch.max(torch.tensor([torch.max(torch.tensor(val_graph_rewards).flatten()), torch.max(torch.tensor(val_means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(val_graph_rewards).flatten()), torch.min(torch.tensor(val_means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(val_means).flatten(), torch.tensor(val_graph_rewards).flatten(), c=torch.tensor(val_graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Val set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_val_example.jpg')


class SupervisedPretrain_GP: #THE NAIVE GP IMPLEMENTATION FOLLOWING PURE MEAN AND COV UPDATED SIMILAT TO 'ALTERNATIVE' IN GNN-UCB

    def __init__(self, target:int=0, input_dim:int=12, reward_plot_dir:str='reward_0', pretrain_indices_name:str='pretrain_indices_rew0', \
                 model_name:str='nnconv_reward0_8000samples_100ep', num_indices:int=7000, laplacian_features=False, laplacian_k=1, std = None, dataset=None):
        
        self.target = target
        self.reward_plot_dir = reward_plot_dir
        self.pretrain_indices_name = pretrain_indices_name
        self.model_name = model_name
        self.laplacian_features = laplacian_features
        self.laplacian_k = laplacian_k
        self.num_indices = num_indices
        self.std = std
        self.dataset = dataset


        if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/'):
            pass
        else:
            os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/')

        self.input_dim = input_dim

        # Split datasets.
        self.subset_indices = np.random.choice(len(self.dataset), self.num_indices, replace=False)
        dataset_subset = dataset[self.subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES

        self.test_dataset = dataset_subset[:int(self.num_indices/10)]
        self.val_dataset = dataset_subset[int(self.num_indices/10):int(self.num_indices/7)]
        self.train_dataset = dataset_subset[int(self.num_indices/7):]
        self.test_loader = DataListLoader(self.test_dataset, batch_size=25, shuffle=False)
        self.val_loader = DataListLoader(self.val_dataset, batch_size=25, shuffle=False)
        self.train_loader = DataListLoader(self.train_dataset, batch_size=25, shuffle=False)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GP(sigma = 1e-4, mean = 0, kernel_features = GNTK_features(GNTK_model=self.GCN) )
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)


    def train_loop(self):
        #self.model.train()
        #self.likelihood.train()
        loss_all = 0
        ct = 0

        for data in self.train_loader:
            #print('Data:', data)
            #print('Len Data:', len(data))
            #print(ct)
            data_cuda = [d.to(self.device) for d in data]
            self.model.fit(data_cuda)
            print('collected:', len(self.model.data['graph_indices']))
            print('G_mat:', self.model.G_mat.shape)
            ct += 1
        print('Count:',ct)
            
    def print_and_plot(self):
    
        #state_dict = self.model.state_dict()
        #torch.save(state_dict, f'/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.model_name}.pt')
        np.save(f'/cluster/scratch/bsoyuer/base_code/graph_BO/{self.pretrain_indices_name}.npy', self.subset_indices)

        plot_loader = DataListLoader(self.train_dataset, batch_size=50, shuffle=False)

        means = []
        graph_rewards = []

        for i, data in enumerate(plot_loader):
            data_cuda = [d.to(self.device) for d in data]
            outputs = self.model.predict(data_cuda)[0]
            #outputs = model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch)
            rewards = [d.y for d in data]

            means.extend(outputs)
            graph_rewards.extend(rewards)

            #print('lean means:', len(means))
            #print('lean rews:', len(graph_rewards))

        max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards).flatten()), torch.max(torch.tensor(means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards).flatten()), torch.min(torch.tensor(means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), c=torch.tensor(graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Training set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_example.jpg')

        plot_loader_val = DataListLoader(self.val_dataset, batch_size=50, shuffle=False)

        val_means = []
        val_graph_rewards = []

       
        for i, data in enumerate(plot_loader_val):
            data_cuda = [d.to(self.device) for d in data]
            outputs = self.model.predict(data_cuda)[0]
            #outputs = model(torch.hstack((data.x, data.pos, data.z[:,None])), data.edge_index, data.batch)
            rewards = [d.y for d in data]

            val_means.extend(outputs)
            val_graph_rewards.extend(rewards)

        max_val = torch.max(torch.tensor([torch.max(torch.tensor(val_graph_rewards).flatten()), torch.max(torch.tensor(val_means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(val_graph_rewards).flatten()), torch.min(torch.tensor(val_means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(val_means).flatten(), torch.tensor(val_graph_rewards).flatten(), c=torch.tensor(val_graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Val set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_val_example.jpg')


class SupervisedPretrain_GP_gpy: #GPYTORCH GP IMPLEMENTATION, OPERATES ON CONCATENATED NODE FEATURES OR MAYBE EXTRACTED NODE FEATURES
    def __init__(self, target:int=0, input_dim:int=12, reward_plot_dir:str='reward_0', pretrain_indices_name:str='pretrain_indices_rew0', \
                 model_name:str='nnconv_reward0_8000samples_100ep', num_indices:int=7000, laplacian_features=False, laplacian_k=1, std = None, num_epochs = 100, dataset=None):
        
        self.target = target
        self.reward_plot_dir = reward_plot_dir
        self.pretrain_indices_name = pretrain_indices_name
        self.model_name = model_name
        self.laplacian_features = laplacian_features
        self.laplacian_k = laplacian_k
        self.num_indices = num_indices
        self.std = std
        self.dataset = dataset
        self.num_epochs = num_epochs


        if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/'):
            pass
        else:
            os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/')

        self.input_dim = input_dim

        # Split datasets.
        self.subset_indices = np.random.choice(len(self.dataset), self.num_indices, replace=False)
        dataset_subset = dataset[self.subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES

        self.test_dataset = dataset_subset[:int(self.num_indices/10)]
        self.val_dataset = dataset_subset[int(self.num_indices/10):int(self.num_indices/7)]
        self.train_dataset = dataset_subset[int(self.num_indices/7):]

        self.test_loader = DataListLoader(self.test_dataset, batch_size=25, shuffle=False)
        self.val_loader = DataListLoader(self.val_dataset, batch_size=25, shuffle=False)
        self.train_loader = DataListLoader(self.train_dataset, batch_size=25, shuffle=False)

        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        self.x_val = []
        self.y_val = []

        for data in self.train_loader:
            self.y_train.extend([d.y for d in data])
            self.x_train.extend([torch.hstack(( feat_pad(d.x), feat_pad(d.pos), z_pad(d.z)[:,None] )).flatten() for d in data] )
        
        self.y_train = torch.tensor(self.y_train)
        self.x_train = torch.stack(self.x_train)

        self.transformer = Normalizer().fit(self.x_train)
        self.x_train = torch.tensor(self.transformer.transform(self.x_train))

        for data in self.val_loader:
            self.y_val.extend([d.y for d in data])
            self.x_val.extend([torch.hstack(( feat_pad(d.x), feat_pad(d.pos), z_pad(d.z)[:,None] )).flatten() for d in data] )
        
        self.y_val = torch.tensor(self.y_val)
        self.x_val = torch.stack(self.x_val)

        for data in self.test_loader:
            self.y_test.extend([d.y for d in data])
            self.x_test.extend([torch.hstack(( feat_pad(d.x), feat_pad(d.pos), z_pad(d.z)[:,None] )).flatten() for d in data] )
        
        self.y_test = torch.tensor(self.y_test)
        self.x_test = torch.stack(self.x_test)


        print('y_train_shape:', self.y_train.shape)
        print('x_train_shape:', self.x_train.shape)

        print('y_val_shape:', self.y_val.shape)
        print('x_val_shape:', self.x_val.shape)

        print('y_test_shape:', self.y_test.shape)
        print('x_test_shape:', self.x_test.shape)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GP_gpy(x_train = self.x_train, y_train = self.y_train, likelihood = self.likelihood).to(self.device)
        #self.model = GP(sigma = 1e-4, mean = 0, kernel_features = GNTK_features(GNTK_model=self.GCN) )
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def return_pretrain_set(self):

        return self.x_train, self.y_train

    def train_loop(self):
        self.model.train()
        self.likelihood.train()
        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        num_epochs = self.num_epochs

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for i in range(num_epochs):

            optimizer.zero_grad()
            output = self.model(self.x_train.to(self.device))
            loss = -self.loss(output, self.y_train.to(self.device))
            loss.backward()
            print('Iter %d - Loss: %.3f   lengthscale: %.3f  noise: %.3f' % (
            i + 1, loss.item(),
            self.model.cov.base_kernel.lengthscale.item(),
            self.model.likelihood.noise.item()))
            optimizer.step()
            scheduler.step()

        return self.model
            
    def print_and_plot(self):
    
        #state_dict = self.model.state_dict()
        #torch.save(state_dict, f'/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.model_name}.pt')
        np.save(f'/cluster/scratch/bsoyuer/base_code/graph_BO/{self.pretrain_indices_name}.npy', self.subset_indices)

        plot_loader = DataListLoader(self.train_dataset, batch_size=50, shuffle=False)

        means = []
        graph_rewards = []

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                
            test_x = torch.tensor(self.x_train).to(self.device)
            observed_pred = self.likelihood(self.model(test_x))
            rewards = self.y_train
            outputs = observed_pred.mean
            print('lean means:', outputs.shape)
            print('lean rews:', rewards.shape)

            means.extend(outputs)
            graph_rewards.extend(rewards)


        max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards).flatten()), torch.max(torch.tensor(means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards).flatten()), torch.min(torch.tensor(means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), c=torch.tensor(graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Training set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_example.jpg')

        plot_loader_val = DataListLoader(self.val_dataset, batch_size=50, shuffle=False)

        val_means = []
        val_graph_rewards = []

       
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            test_x = torch.tensor(self.transformer.transform(self.x_test)).to(self.device)
            observed_pred = self.likelihood(self.model(test_x))
            rewards = self.y_test
            outputs = observed_pred.mean
            #print('lean means:', means.shape)
            #print('lean rews:', graph_rewards.shape)

            val_means.extend(outputs)
            val_graph_rewards.extend(rewards)

        max_val = torch.max(torch.tensor([torch.max(torch.tensor(val_graph_rewards).flatten()), torch.max(torch.tensor(val_means).flatten())]))
        min_val = torch.min(torch.tensor([torch.min(torch.tensor(val_graph_rewards).flatten()), torch.min(torch.tensor(val_means).flatten())]))
        #print(max_val)

        plt.figure()
        plt.scatter(torch.tensor(val_means).flatten(), torch.tensor(val_graph_rewards).flatten(), c=torch.tensor(val_graph_rewards).cpu(), cmap="gist_ncar", s=1/2)
        plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
        plt.title('Val set')
        plt.colorbar()
        plt.legend()
        plt.xlabel('Predicted')
        plt.ylabel("True Reward")
        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{self.reward_plot_dir}/pyg_demo_2_val_example.jpg')

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(self.transformer.transform(self.x_val[:self.y_val.shape[0],:])).to(self.device)
            observed_pred = self.likelihood(self.model(test_x))

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(np.arange(self.y_val.shape[0]), self.y_train[:self.y_val.shape[0]].numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(np.arange(self.y_val.shape[0]), observed_pred.mean.cpu().numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(np.arange(self.y_val.shape[0]), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

            f.savefig('gp_results.jpg')

class GP_minimal(nn.Module): #THE MINIMAL GP IMPLEMENTATION ON GITHUB, BRIDGING SCIKIT-LEARN MEAN AND COV UPDATES AND HYPERPARAM OPT. IN GPYTORCH
    def __init__(self, length_scale=1.0, noise_scale=1.0, amplitude_scale=1.0):
        super().__init__()
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)

    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        y = self.y
        L = self.L
        alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = k.T.mm(alpha)
        var = self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v))
        return mu, var

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        D = X.shape[1]
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = (
            -0.5 * y.T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        )
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        return marginal_likelihood

    def kernel_mat_self(self, X):
        sq = (X**2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 * X.mm(X.T)
        return self.amplitude_scale * torch.exp(
            -0.5 * sqdist / self.length_scale
        ) + self.noise_scale * torch.eye(len(X))

    def kernel_mat(self, X, Z):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return self.amplitude_scale * torch.exp(-0.5 * sqdist / self.length_scale)

    def train_step(self, X, y, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        nll = -self.fit(X, y).sum()
        nll.backward()
        opt.step()
        return {
            "loss": nll.item(),
            "length": self.length_scale.detach().cpu(),
            "noise": self.noise_scale.detach().cpu(),
            "amplitude": self.amplitude_scale.detach().cpu(),
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNN-UCB run')

    # environment arguments
    # this is to set which dataset to pick

    parser.add_argument('--num_nodes', type=int, default=5, help = 'max number of nodes per graph')
    parser.add_argument('--feat_dim', type = int, default=15, help ='Dimension of node features for the graph')
    parser.add_argument('--edge_prob', type=float, default=0.05, help='probability of existence of each edge, shows sparsity of the graph')
    parser.add_argument('--data_size', type=int, default=5, help = 'size of the seed dataset for generating the reward function')
    parser.add_argument('--noise_var', type=float, default=0.0001, help = 'variance of noise for observing the reward, if exists')
    parser.add_argument('--seed', type=int, default=354)
    parser.add_argument('--nn_init_lazy', type=bool, default=True)
    parser.add_argument('--exp_result_folder', type=str, default="./results/hyperparamgnnucb_bartu/")
    parser.add_argument('--print_every', type=int, default=20)
    #parser.add_argument('--runner_verbose', type=bool, default=True)
    parser.add_argument('--runner_verbose', default=False, action='store_true', help='Bool type')

    # model arguments
    parser.add_argument('--net', type=str, default='GNN', help='Network to use for UCB')
    parser.add_argument('--noisy_reward', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_mlp_layers_alg', type=int, default=2)
    #parser.add_argument('--train_from_scratch', type=bool, default=True)
    parser.add_argument('--train_from_scratch', default=False, action='store_true', help='Bool type')
    parser.add_argument('--pretrain_steps', type=int, default=40)
    parser.add_argument('--t_intersect', type = int, default=100)
    parser.add_argument('--neuron_per_layer', type=int, default=2048)
    parser.add_argument('--exploration_coef', type=float, default=0.0098) #0.0098
    #parser.add_argument('--alg_lambda', type=float, default=0.0063)
    parser.add_argument('--alg_lambda', type=float, default=0.0063)
    parser.add_argument('--nn_aggr_feat', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--T', type=int, default=320)
    parser.add_argument('--T0', type=int, default=100)

    parser.add_argument('--data', type=str, default='QM9DATA/QM9_w_edgeix/', help='dataset type')
    #parser.add_argument('--synthetic', action='store_true') #If you dont specify synthetic, False, if you put only '--synthetic', True
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--dataset_size', type=int, default=130831)
    parser.add_argument('--num_actions', type=int, default=100, help = 'size of the actions set, i.e. total number of graphs')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help = 'number of MLP layer for the GNTK that creates the synthetic data')

    parser.add_argument('--stop_count', type=int, default=1000)
    parser.add_argument('--relative_improvement', type=float, default=1e-4)
    parser.add_argument('--small_loss', type=float, default=1e-3)

    parser.add_argument('--load_pretrained', default=False, action='store_true', help='Bool type')
    parser.add_argument('--pretrain_model_name', type=str, default='nnconv_reward0_8000samples_100ep')
    parser.add_argument('--pretrain_indices_name', type=str, default='pretrain_indices')
    parser.add_argument('--pretrain_num_indices', type=int, default=7000)

    parser.add_argument('--explore_threshold', type=int, default=2)

    parser.add_argument('--T1', type=int, default=80, help="Number of steps (from step 0) until the network is trained fom scratch")

    parser.add_argument('--T2', type=int, default=180, help="To test pure exploration  to decouple overfitting and exploration issues")

    parser.add_argument('--dropout', default=False, action='store_true', help='Bool type')

    parser.add_argument('--dropout_prob', type=float, default=0.2)

    parser.add_argument('--subsample', default=False, action='store_true', help='subsamples from the graph indices explored so far to be used in GD')
    parser.add_argument('--subsample_method', default='random', choices=['random','weights','inverse_weights'])
    parser.add_argument('--subsample_num', type=int, default=20)

    parser.add_argument('--pool', default=False, action='store_true', help='Whether to use batching where the algorithm acts on pooled data')
    parser.add_argument('--pool_num', type=int, default=20)
    parser.add_argument('--greedy', default=False, action='store_true', help='Whether to greedily pick maximum mean pt on pooled pts')

    parser.add_argument('--online_cov', default=False, action='store_true', help='Compute inverse kernel matrix via online updates depending on inverses of txt matrices')
    parser.add_argument('--complete_cov_mat', default=False, action='store_true', help='Use complete (not diagonal approx) cov matrix')
    parser.add_argument('--alternative', default=False, action='store_true', help='Compute Mahalanobis approx to variance without online updates')
    
    parser.add_argument('--GD_batch_size', type=int, default=10, help="Batch Size in batch GD")
    parser.add_argument('--batch_GD', default=False, action='store_true', help='Whether to train using batched samples in GD')
    
    parser.add_argument('--factor', type=float, default=0.7, help='LR Scheduler factor')
    parser.add_argument('--patience', type=int, default=5, help='LR Scheduler patience')
    parser.add_argument('--dim', type=int, default=64, help='Dim for NNConv')

    parser.add_argument('--no_var_computation', default=False, action='store_true', help='Whether to not bother with computing confidences')

    parser.add_argument('--batch_window', default=False, action='store_true', help='Whether to use a window sliging over collected incices to take alst batch_window_size samples for GD')
    parser.add_argument('--batch_window_size', type=int, default=80)

    parser.add_argument('--focal_loss', default=False, action='store_true', help='Whether to use weighted focal loss')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=2.0)

    parser.add_argument('--large_scale', default=False, action='store_true', help='Whether you are running lergescale exp with many graphs, adjusts saving location of plots')

    parser.add_argument('--remove_pretrain', default=False, action='store_true', help='Whether to remove dataset indices used for pretraining in the beginning')

    parser.add_argument('--reward', type=int, default=0, help='Choose the index of the reward to deploy the algorithm on, in [0,18]')

    parser.add_argument('--bernoulli_selection', default=False, action='store_true', help='Whether to apply UCB with with a coin flip whose parameter decays with T, otherwise greedy')

    parser.add_argument('--ucb_wo_replacement', default=False, action='store_true', help='Whether to apply UCB without replacement')

    parser.add_argument('--reward_plot_dir', type=str, default='reward_0', help='Subdir according to reward to save results in')

    parser.add_argument('--pool_top_means', default=False, action='store_true', help='Pool -pool_num- many datapts by selecting pts with top predicted means, update these means in each batch_size steps')

    parser.add_argument('--small_net_var', default=False, action='store_true', help='Use a smaller GCN to compute gradient features and posterior variances')

    parser.add_argument('--initgrads_on_fly', default=False, action='store_true', help='Compute gradient featues on the fly only when necessary')

    parser.add_argument('--oracle', default=False, action='store_true', help='The agent is replaced by an oracle')

    parser.add_argument('--select_K_together', default=False, action='store_true', help='Whether to select K samples at a time in each step')
    parser.add_argument('--select_K', type=int, default=5, help='Number of samples to select simultaneously')

    parser.add_argument('--laplacian_features', default=False, action='store_true', help='Whether to concatenate topk laplacian eigenvectors to node features')
    parser.add_argument('--laplacian_k', type=int, default=1)

    parser.add_argument('--pretraining_load_pretrained', default=False, action='store_true', help='Whether to load a pretrained model on another reward during supervised pretraining for transfer learning purpsoes')
    parser.add_argument('--pretraining_pretrain_model_name', type=str, default='nnconv_reward3n4_8000samples_100ep')

    args = parser.parse_args()

    args.T = args.T if not args.select_K_together else int(float(args.T) / float(args.select_K))
    args.T0 = args.T0 if not args.select_K_together else int(float(args.T0) / float(args.select_K))
    args.T1 = args.T1 if not args.select_K_together else int(float(args.T1) / float(args.select_K))
    args.T2 = args.T2 if not args.select_K_together else int(float(args.T2) / float(args.select_K))
    args.pretrain_steps = args.pretrain_steps if not args.select_K_together else int(float(args.pretrain_steps) / float(args.select_K))
    args.print_every = args.print_every if not args.select_K_together else int(float(args.print_every) / float(args.select_K))
    args.batch_size = args.batch_size if not args.select_K_together else int(float(args.batch_size) / float(args.select_K))

    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))

    print('Args:',args)

    env_rds = np.random.RandomState(args.seed)
    env_rds_choice = np.random.Generator(np.random.PCG64(args.seed)) #Gnumpy random Generator which is supposed to be faster

    target = args.reward

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')

    if args.laplacian_features:
        transform = T.Compose([MyTransform(target), T.AddLaplacianEigenvectorPE(k=args.laplacian_k, attr_name=None, is_undirected=True), Complete(), T.Distance(norm=False)])
        args.feat_dim += args.laplacian_k
    else:
        transform = T.Compose([MyTransform(target), Complete(), T.Distance(norm=False)])

    dataset = QM9(path, transform=transform)

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

    pretraining_alg = SupervisedPretrain_GP_gpy(input_dim = args.feat_dim, reward_plot_dir = args.reward_plot_dir, \
                                             pretrain_indices_name=args.pretrain_indices_name, model_name=args.pretrain_model_name, num_indices=args.pretrain_num_indices, \
                                             laplacian_features=args.laplacian_features, laplacian_k=args.laplacian_k, std = std, dataset=dataset)
    pretraining_alg.train_loop()
    pretraining_alg.print_and_plot()

    # train_x = torch.stack((torch.linspace(0, 1, 100), torch.linspace(0, 1, 100)), dim = 1)
    # print(train_x.shape)
    # # True function is sin(2*pi*x) with Gaussian noise
    # train_y = torch.sin(train_x[:,0] * (2 * math.pi)) + torch.cos(train_x[:,1] * (2 * math.pi)) + torch.randn(train_x.shape[0]) * math.sqrt(0.04)
    # print(train_y.shape)

    # pretraining_alg = SupervisedPretrain_GP_gpy(input_dim = args.feat_dim, reward_plot_dir = args.reward_plot_dir, \
    #                                          pretrain_indices_name=args.pretrain_indices_name, model_name=args.pretrain_model_name, num_indices=args.pretrain_num_indices, \
    #                                          laplacian_features=args.laplacian_features, laplacian_k=args.laplacian_k, std = 1, dataset=dataset)
    # pretraining_alg.train_loop()
    # pretraining_alg.print_and_plot()
    