import numpy as np
import sys 
import os
sys.path.append(os.path.abspath("./base_code/graph_BO"))
sys.path.append(os.path.abspath("./"))
import torch.optim as optim
import copy
from nets import NN,GNN
from nets_w_edgeix import NNConv_Net as GNN_batched
#from nets_w_edgeix import GNN_pyg as GNN_batched
#from nets_temp import GNN_pyg as GNN_init_grad
#from nets_temp import GNN_old as GNN_init_grad
from nets_temp import GNN_pyg as GNN_init_grad
from nets_temp import normalize_init
from nets_batched import NN as NN_batched
from typing import Optional
from config import *
from loss import FocalLoss
from torch_geometric.loader import DataLoader
import time 
import collections.abc
import torch.multiprocessing as mp

"""EXPLORATION COEF: BETA
LAMBDA: MSE REGULARIZATION, ALSO THE ALEOTORIC NOISE
COMPELTE COV MAT: COMPUTE POSTERIOR MATRICERS FULLY OR APPROX BY DIAGONALS
RANDOM STATE: RANDOM SEED
TRAIN FROM SCRATCH: ?"""

def handle_error(error):
    print(error, flush = True)

class UCBalg:
    def __init__(self, qm9_data, qm9_val_data, init_grad_data, init_grad_loader, dataset_loader, val_dataset_loader, mean: float, std: float,
                 net: str, feat_dim: int, num_nodes: int, dim: int, batch_GD: bool, num_mlp_layers: int = 1, alg_lambda: float = 1,
                 exploration_coef: float = 1, neuron_per_layer: int = 100,
                 complete_cov_mat: bool = False, lr: float = 1e-3,
                 random_state = None, nn_aggr_feat: bool = True, no_var_computation: bool = False,
                 train_from_scratch=False, stop_count=1000, relative_improvement=1e-4, small_loss = 1e-3, patience = 5, factor = 0.7, reward = 0,
                 load_pretrained=False, pretrain_model_name=None, dropout=False, dropout_prob=0.2, subsample=False, subsample_method='random', verbose = True,
                 subsample_num=20, greedy=False, online=False, alternative = False, GD_batch_size=20, pool=False, pool_num=20, batch_window=False, batch_window_size=80, focal_loss=False, 
                 alpha=0.25, gamma=2.0, large_scale=False, bernoulli_selection=False, ucb_wo_replacement=False, pool_top_means=False, small_net_var = False, initgrads_on_fly = False, oracle=False, select_K_together=False, select_K=5, batch_size = 50, thompson_sampling = False, path: Optional[str] = None, **kwargs):

        self.GD_batch_size = GD_batch_size

        self.small_net_var = small_net_var

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', self.device)
        print('Name:', torch.cuda.get_device_name(self.device))

        self.device_select = torch.device('cpu')
        
        if batch_GD:
            print('Using the nets from nets_w_edgeix module')
            if net == 'NN': #TO BE ABLE TO COMPARE AGAINST NAIVE NTK
                self.func = NN_batched(input_dim=feat_dim * num_nodes, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(self.device)
            elif net == 'GNN':
                if self.small_net_var:
                    print('Using 1 layer net for init grads')
                    #self.func_init_grad = GNN_init_grad(input_dim=15, depth=1, width=256, dropout=False, dropout_prob=0.2, batch_size=self.GD_batch_size, aggr_feats=True).to(self.device).float()
                    self.func_init_grad = GNN_init_grad(input_dim=feat_dim+3, depth=1, width=256, dropout=False, dropout_prob=0.2, aggr_feats=False).to(self.device).float()
                else:
                    print('Using actual network for NTK Computations')
                    self.func_init_grad = GNN_batched(input_dim=feat_dim, dim=dim, width=neuron_per_layer, aggr_feats=False).to(self.device).float()
                self.func = GNN_batched(input_dim=feat_dim, dim=dim, width=neuron_per_layer, aggr_feats=False ).to(self.device).float()
            else:
                raise NotImplementedError
        else:
            if net == 'NN': #TO BE ABLE TO COMPARE AGAINST NAIVE NTK
                self.func = NN(input_dim=feat_dim * num_nodes, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(self.device)
            elif net == 'GNN':
                self.func = GNN(input_dim=feat_dim, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat, dropout=dropout, dropout_prob=dropout_prob).to(self.device)
            else:
                raise NotImplementedError
            
        self._rds = np.random if random_state is None else random_state
        self.alg_lambda = alg_lambda  # lambda regularization for the algorithm, FOR MSE LOSS
        #self.num_net_params = sum(p.numel() for p in self.func.parameters() if p.requires_grad) #COUNT ALL MODEL PARAMS THAT REQUIRE GRAD, I.E. TRAINABLE
        self.num_net_params = sum(p.numel() for p in self.func_init_grad.parameters() if p.requires_grad)
        self.U = torch.zeros((self.num_net_params,)).to(self.device_select)
        self.U_inv_small = None
        self.exploration_coef = exploration_coef
        self.neuron_per_layer = neuron_per_layer
        self.train_from_scratch = train_from_scratch

        self.complete_cov_mat = complete_cov_mat  # if true, the complete covariance matrix is considered. Otherwise, just the diagonal.
        self.lr = lr

        self.stop_count = stop_count
        self.relative_improvement = relative_improvement
        self.small_loss = small_loss

        self.load_pretrained = load_pretrained

        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.subsample = subsample
        self.subsample_method = subsample_method
        self.subsample_num = subsample_num

        self.greedy = greedy

        self.online = online

        self.alternative = alternative

        self.verbose = verbose

        self.epsilon_var = 1e-4
        self.epsilon_diag = 1e-6
        self.dtype = torch.float64

        self.QM9_Dataset = qm9_data
        self.QM9_Val_Dataset = qm9_val_data
        self.QM9_Dataset_init_grad = init_grad_data

        self.collected_indices = []

        self.dataloader_init_grad = init_grad_loader
        self.dataloader_dataset = dataset_loader
        self.dataloader_val_dataset = val_dataset_loader

        self.batch_GD = batch_GD

        self.dim = dim

        self.mean = mean
        self.std = std

        self.patience = patience
        self.factor = factor

        self.pool = pool
        self.pool_num  = pool_num

        self.no_var_computation = no_var_computation

        self.batch_window = batch_window
        self.batch_window_size = batch_window_size
        self.effective_batch_window_size = None

        self.focal_loss = focal_loss
        self.alpha=alpha
        self.gamma=gamma

        self.large_scale=large_scale   

        self.pretrain_model_name = pretrain_model_name 

        self.bernoulli_selection = bernoulli_selection  

        self.ucb_wo_replacement = ucb_wo_replacement

        self.pool_top_means = pool_top_means

        self.initgrads_on_fly = initgrads_on_fly

        self.oracle = oracle 

        self.reward = reward

        self.select_K_together = select_K_together
        self.select_K = select_K

        self.batch_size = batch_size

        self.thompson_sampling = thompson_sampling

        if path is None: #BELOW JUST PUTS THE SPECIFIED PARAMS INSIDE THE CURLY BRACES IN RESPECTIVE ORDER
            self.path = 'trained_models/{}_{}dim_{}L_{}m_{:.3e}beta_{:.1e}sigma'.format(net, feat_dim, num_mlp_layers,
                                                                                        neuron_per_layer,
                                                                                        self.exploration_coef,
                                                                                        self.alg_lambda)
        else:
            self.path = path

        self.G = None

        self.Kt_inv = None
        self.b = None
        self.K11 = None
        self.K12 = None
        self.K21 = None
        self.K22 = None

    def get_infogain(self): #MUTUAL INFO BETWEEN PREVIOSLY SEEN G1,...,GT AND GNN OUTPUT F_GNN. KERNEL MATRIX HERE IS KGNN_T (EQ5 IN PAPER)
        kernel_matrix = self.U - self.alg_lambda * torch.ones((self.num_net_params,)).to(device)
        return 0.5 * np.log(torch.prod(1 + kernel_matrix / self.alg_lambda).cpu().numpy()) #DETERMINANT SIMPLY BERCOMES PRODUCT AS WE 
        #REPRESENT THESE DIAGONALIZED KERNEL MATRICES WITH VECTORS

    def save_model(self):
        torch.save(self.func, self.path)

    def load_model(self):
        try:
            self.func = torch.load(self.path)
            self.func.eval() #LOAD TRAINED MODEL AND SET TO EVALUATION MODE
        except:
            print('Pretrained model not found.')

class GnnUCB(UCBalg):  # Our main method
    # This class currently uses Woodbury's Identity(EQ8 IN PAPER). For scalability experiment, we need to use the regular gradient.
    def __init__(self, qm9_data, qm9_val_data, init_grad_data, init_grad_loader, dataset_loader, val_dataset_loader,
                 mean: float, std: float, net: str,num_nodes: int, feat_dim: int, dim: int, num_actions: int, action_domain: list, batch_GD: bool,
                 alg_lambda: float = 1, exploration_coef: float = 1, t_intersect: int = np.inf,
                 num_mlp_layers: int = 2, neuron_per_layer: int = 128, lr: float = 1e-3, no_var_computation = False, reward = 0,
                 nn_aggr_feat = True, train_from_scratch = False, stop_count=1000, relative_improvement=1e-4, small_loss = 1e-3, load_pretrained=False, pretrain_model_name=None,
                 dropout=False, dropout_prob=0.2, subsample=False, subsample_method='random', subsample_num=20, greedy=False, online=False, verbose = True,
                 nn_init_lazy: bool = True, complete_cov_mat: bool = False, alternative=False, GD_batch_size=10, patience = 5, factor = 0.7, pool=False, pool_num=20, batch_window=False, batch_window_size=80, focal_loss = False,
                 alpha=0.25, gamma=2.0, large_scale=False, bernoulli_selection=False, ucb_wo_replacement=False, pool_top_means=False, initgrads_on_fly = False, oracle=False, select_K_together=False, select_K=5, batch_size = 50, thompson_sampling = False, random_state = None, path: Optional[str] = None, **kwargs):
        super().__init__(qm9_data=qm9_data, qm9_val_data=qm9_val_data, init_grad_data=init_grad_data, init_grad_loader=init_grad_loader, dataset_loader=dataset_loader, val_dataset_loader=val_dataset_loader, net=net, feat_dim=feat_dim, dim=dim, num_mlp_layers=num_mlp_layers, alg_lambda=alg_lambda, verbose = verbose, #INHERITS THE BASE UCB ALGO CLASS
                         mean=mean, std=std, lr = lr, complete_cov_mat = complete_cov_mat, nn_aggr_feat = nn_aggr_feat, no_var_computation = no_var_computation, train_from_scratch = train_from_scratch, 
                         stop_count=stop_count, relative_improvement=relative_improvement, small_loss=small_loss, load_pretrained=load_pretrained, pretrain_model_name=pretrain_model_name, num_nodes=num_nodes,
                         exploration_coef=exploration_coef, dropout_prob=dropout_prob, dropout=dropout, neuron_per_layer=neuron_per_layer, random_state=random_state, reward = reward,
                         subsample=subsample, subsample_method=subsample_method, subsample_num=subsample_num, greedy=greedy, online=online, alternative=alternative, GD_batch_size=GD_batch_size, 
                         batch_GD=batch_GD, factor = factor, patience = patience, pool=pool, pool_num=pool_num, batch_window = batch_window, batch_window_size=batch_window_size, focal_loss=focal_loss, 
                         alpha=alpha, gamma=gamma, large_scale=large_scale, bernoulli_selection=bernoulli_selection, ucb_wo_replacement=ucb_wo_replacement, pool_top_means=pool_top_means, initgrads_on_fly = initgrads_on_fly, 
                         select_K_together=select_K_together, select_K=select_K, oracle=oracle, batch_size = batch_size, thompson_sampling = thompson_sampling, path=path, **kwargs)

        self.nn_aggr_feat = nn_aggr_feat
        # Create the network for computing gradients and subsequently variance.
        if self.load_pretrained:
            #self.func = torch.load('/local/bsoyuer/base_code/graph_BO/results/saved_models/reward1_5epochs')
            #self.func.load_state_dict(torch.load('/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/nnconv_reward0_1000samples_100ep.pt'))
            self.func.load_state_dict(torch.load(f'/cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.pretrain_model_name}.pt'))
            self.func.train()
            self.f0 = copy.deepcopy(self.func)
            self.f0 = normalize_init(self.f0)
            self.func_init_grad = normalize_init(self.func_init_grad)
            print(f"Loaded Pretrained Model From /cluster/scratch/bsoyuer/base_code/graph_BO/results/saved_models/{self.pretrain_model_name}.pt")
        else:
            self.f0 = copy.deepcopy(self.func) #SO THAT AS NETWORK FUNCTION IS UPDATED, INITIALIZATION ISNT!
            self.f0 = normalize_init(self.f0)
            self.func_init_grad = normalize_init(self.func_init_grad)

            if nn_init_lazy:
                #self.func = normalize_init(self.func)
                self.f0 = copy.deepcopy(self.func)
                self.f0 = normalize_init(self.f0)
                
                self.func_init_grad = normalize_init(self.func_init_grad)

        if net == 'NN':
            self.name = 'NN-UCB'
        else:
            self.name = 'GNN-UCB'

        self.data = {
            'graph_indices': [], #GRAPH DATA
            'rewards': [], #CONTINOUS REWARD
            'weights': [],
            'means': [] #PREDICTED MEANS FOR EACH SAMPLE IN THE DATASET, UPDATED ONCE EVERY BATCH_SIZE STEPS!!
        }

        self.unique_data = {   #ONLY ADD IF PT NOT OBSERVED BEFORE
            'graph_indices': [],
            'ucb_replacement_graph_indices': [],
            'rewards': [],
            'weights': []
        }

        self.num_actions = num_actions #Actıon set sıze
        self.action_domain = action_domain #List, Value Inputted In Other Code?

        self.init_grad_list = []
        self.init_grad_list_cpu = []
        t_0 = time.time()
        if not self.initgrads_on_fly:
            self.get_init_grads()
        t_1 = time.time()
        print('Time for Getting Init Grads:', t_1-t_0)

    def save_model(self):
        super().save_model()
        torch.save(self.f0, self.path + "/f0_model")

    def init_grads_on_demand(self,indices):
        #print(indices)
        if not isinstance(indices, collections.abc.Sequence):
            indices_list = [indices]
        else:
            indices_list = indices
        print('indices_list:',indices_list)
        post_mean0 = []
        init_grads_on_demand = []
        #print(indices_list)
        #print(data)

        if len(indices_list) == 1:
            data = self.QM9_Dataset_init_grad[indices_list]
            loader = DataLoader(data, batch_size=1, shuffle=False)
            for i, d in enumerate(data):
                self.func_init_grad.zero_grad()
                post_mean0.append(self.func_init_grad(d.to(self.device)))
                post_mean0[-1].backward(retain_graph=True)
                g = torch.cat([p.grad.flatten().detach() for p in self.func_init_grad.parameters()])
                init_grads_on_demand.append(g)

            return g
        
        else:
            data = self.QM9_Dataset_init_grad[indices_list]
            loader = DataLoader(data, batch_size=1, shuffle=False)
            for i, d in enumerate(loader):
                self.func_init_grad.zero_grad()
                post_mean0.append(self.func_init_grad(d.to(self.device)))
                post_mean0[-1].backward(retain_graph=True)
                g = torch.cat([p.grad.flatten().detach() for p in self.func_init_grad.parameters()])
                init_grads_on_demand.append(g)

            return torch.stack(init_grads_on_demand)
            

    def get_init_grads(self):
        post_mean0 = []
        if self.batch_GD:
            for i, data in enumerate(self.dataloader_init_grad):
                #print(data)
                #self.f0.zero_grad() #Clear The gradients computed by bacxkwards for previous Graph in domain
                self.func_init_grad.zero_grad()
                post_mean0.append(self.func_init_grad(data.to(self.device)))
                post_mean0[-1].backward(retain_graph=True) #Compute gradients wrt the last forward pass,
                #since we do multiple backwards on the same computational graph (with gradients cleared at each step) iteratively,
                #so we dont want the implicit computattions in the networw pretaining to f0 to be freed!!!
                # Get the Variance.
                #g = torch.cat([p.grad.flatten().detach() for p in self.f0.parameters()]) #Backward computes grads' .grad method accesses them
                g = torch.cat([p.grad.flatten().detach() for p in self.func_init_grad.parameters()])
                #Flatten the gradients wrt each parameter anc concatenates them end to end to gett a full gradient vecotr, g_theta(Graph)
                self.init_grad_list.append(g)
                self.init_grad_list_cpu.append(g.cpu())
            self.init_grad_list_cpu = torch.stack(self.init_grad_list_cpu).share_memory_()
            #print('SELF INIT GRADS CPU:', self.init_grad_list_cpu)
        else:
            for graph in self.action_domain:
                self.f0.zero_grad() #Clear The gradients computed by bacxkwards for previous Graph in domain
                post_mean0.append(self.f0(graph)) #Algortihm uses GNN output(but not at init?) as mean estimate
                post_mean0[-1].backward(retain_graph=True) #Compute gradients wrt the last forward pass,
                #since we do multiple backwards on the same computational graph (with gradients cleared at each step) iteratively,
                #so we dont want the implicit computattions in the networw pretaining to f0 to be freed!!!
                # Get the Variance.
                g = torch.cat([p.grad.flatten().detach() for p in self.f0.parameters()]) #Backward computes grads' .grad method accesses them
                #Flatten the gradients wrt each parameter anc concatenates them end to end to gett a full gradient vecotr, g_theta(Graph)
                self.init_grad_list.append(g)
        print(self.init_grad_list[0].shape)
            

    # def get_small_cov(self, g: np.ndarray):
    #     # Need to check square root. In any case, it is an issue of scaling - sweeping over beta properly would work.
    #     k_xx = g.dot(g)
    #     k_xy = torch.matmul(g.reshape(1, -1), self.G.T)
    #     k_xy = torch.matmul(k_xy, self.U_inv_small)
    #     k_xy = torch.matmul(k_xy, torch.matmul(self.G, g.reshape(-1, 1)))
    #     final_val = k_xx - k_xy
    #     return final_val

    def add_data_ucb_replacement(self, indices, rewards):
        for idx, reward in zip(indices, rewards):
            if idx not in self.unique_data['ucb_replacement_graph_indices']:
                self.unique_data['ucb_replacement_graph_indices'].append(idx)

    def add_data(self, indices, rewards):
        # add the new observations, only if it didn't exist already
        #print("Shape:", np.array(self.init_grad_list).shape)
        for idx, reward in zip(indices, rewards):
            #if idx not in self.data['graph_indices']: #TODO: uncomment?
            self.data['graph_indices'].append(idx)
            self.data['rewards'].append(reward)

            self.collected_indices.append(idx)

            '''
            ALSO, KEEP TRACK OF THE UNIQUELY COLLECTED PTS AND THEIR WEIGHTS WHICH ARE INVERSELY PROPORTIONAL TO THE 
            VALUE OF THE STEP IN WHICH THEY WERE ACQUIRED FIRST
            '''
            if idx not in self.unique_data['graph_indices']:
                self.unique_data['graph_indices'].append(idx)
                #self.unique_data['rewards'].append(reward)
                #self.unique_data['weights'] = np.flip(np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices']))))
                #self.unique_data['weights'] = np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices'])))
            
            if self.no_var_computation:
                print('Post Vars will not be computed')
                pass
            else:
                if self.online:

                    print("Computing inverse covariance online")
                    #print(len(self.data['graph_indices']))
                    g_to_add = self.init_grad_list[idx].to(self.device)
                    #print(g_to_add.shape)

                    #if self.G is None:
                        #self.G = g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)
                    #else:
                        #self.G = torch.cat((self.G, g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)), dim=0)
                    
                    #kernel_matrix = torch.matmul(self.G, self.G.t())

                    #U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]) * self.alg_lambda) + kernel_matrix)

                    if self.Kt_inv is None:
                        self.Kt_inv = 1 / ((torch.sum(g_to_add.reshape(1, -1).to(dtype=self.dtype) * g_to_add.reshape(1, -1).to(dtype=self.dtype), dim=1)/self.neuron_per_layer) + self.alg_lambda)
                        #print("Kt_inv:", self.Kt_inv.shape)
                    else:
                        if self.Kt_inv.dim() <= 1: 
                            #print("tensor:", torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices'][:-1]].flatten(start_dim = 1).shape)
                            #print("gtoadd:", g_to_add.shape)
                            #print("Kt_inv:", self.Kt_inv.shape)
                            #self.b = np.array([self.init_grad_list[ix] for ix in self.data['graph_indices']])
                            #self.b = torch.sum(torch.stack(self.init_grad_list)[self.data['graph_indices'][:-1]].flatten(start_dim = 1) * \
                                #torch.stack(self.init_grad_list)[self.data['graph_indices'][:-1]].flatten(start_dim = 1), dim=1) / self.neuron_per_layer  
                            self.b = (torch.matmul(g_to_add.to(dtype=self.dtype), \
                                torch.stack(self.init_grad_list, dim = 1).to(dtype=self.dtype)[:,self.data['graph_indices'][:-1]].flatten(start_dim = 1)).to(dtype=self.dtype) / self.neuron_per_layer).reshape((-1,1))
                            #print("b:", self.b.shape)
                            b_transpose = torch.t(self.b)  
                            #print("b_transpose:", b_transpose.shape)   
                            #Kt = 1/self.Kt_inv          
                            self.K22 = 1 / (torch.sum(g_to_add.reshape(1, -1).to(dtype=self.dtype) * g_to_add.reshape(1, -1).to(dtype=self.dtype), dim=1)/self.neuron_per_layer + self.alg_lambda - b_transpose * self.Kt_inv *self.b)
                            #print("K22:", self.K22.shape)
                            self.K11 = self.Kt_inv + self.K22 * self.Kt_inv * self.b * b_transpose * self.Kt_inv
                            #print("K11:", self.K11.shape)
                            self.K12 = - self.K22 * self.Kt_inv * self.b
                            #print("K12:", self.K12.shape)
                            self.K21 = - self.K22 * b_transpose * self.Kt_inv
                            #print("K21:", self.K21.shape)
                            #self.Kt_inv = torch.cat((torch.cat((self.K11[:,None], self.K12[:,None]), dim=1),torch.cat((self.K21[:,None], self.K22[:,None]), dim=1)), dim=0)
                            self.Kt_inv = torch.vstack((torch.hstack((self.K11, self.K12)),torch.hstack((self.K21, self.K22)))).to(dtype=self.dtype)
                            #print("Kt_inv:", self.Kt_inv.shape)
                        else: 
                            #print("Kt_inv:", self.Kt_inv.shape)
                            #self.b = np.array([self.init_grad_list[ix] for ix in self.data['graph_indices']])
                            self.b = (torch.matmul(g_to_add.to(dtype=self.dtype), \
                                torch.stack(self.init_grad_list, dim = 1).to(dtype=self.dtype)[:,self.data['graph_indices'][:-1]].flatten(start_dim = 1)).to(dtype=self.dtype) / self.neuron_per_layer).reshape((-1,1))
                            #print("b:", self.b.shape)
                            b_transpose = torch.t(self.b) 
                            #print("b_transpose:", b_transpose.shape)   
                            #Kt = torch.linalg.inv(self.Kt_inv)           
                            self.K22 = 1 / (torch.sum(g_to_add.reshape(1, -1).to(dtype=self.dtype) * g_to_add.reshape(1, -1).to(dtype=self.dtype))/self.neuron_per_layer + self.alg_lambda - torch.matmul(torch.matmul(b_transpose, self.Kt_inv), self.b))
                            #print("K22:", self.K22.shape)
                            self.K11 = self.Kt_inv + torch.matmul(torch.matmul(self.K22 * self.Kt_inv, torch.matmul(self.b, b_transpose)), self.Kt_inv) 
                            #print("K11:", self.K11.shape)
                            self.K12 = - torch.matmul(self.K22 * self.Kt_inv, self.b) #I THINK MISWRITTEN IN PAPER, SHOULDNT BE KT
                            #print("K12:", self.K12.shape)
                            self.K21 = - torch.matmul(self.K22 * b_transpose, self.Kt_inv)
                            #print("K21:", self.K21.shape)
                            #self.Kt_inv = torch.cat((torch.cat((self.K11, self.K12[:,None]), dim=1),torch.cat((self.K21[:,None].t(), self.K22[:,None]), dim=1)), dim=0)
                            self.Kt_inv = torch.vstack((torch.hstack((self.K11, self.K12)),torch.hstack((self.K21, self.K22)))).to(dtype=self.dtype)
                            #print("Kt_inv:", self.Kt_inv.shape)
                    #if len(self.data['graph_indices']) <= 4:
                        #print('U:',U)
                        #print('Kt inv:', self.Kt_inv)
                    #print('Equal:', torch.eq(U, self.Kt_inv))
                #g_to_add = self.init_grad_list[idx]
                elif self.complete_cov_mat:
                    print("Using complete cov matrix")
                    g_to_add = self.init_grad_list[idx].to(self.device)
                    #raise NotImplementedError
                    if self.G is None:
                        self.G = torch.tensor(g_to_add.reshape(1, -1)).to(dtype=self.dtype).to(device) / np.sqrt(self.neuron_per_layer)
                    else:
                        self.G = torch.cat((self.G, g_to_add.reshape(1, -1).to(dtype=self.dtype).to(device) / np.sqrt(self.neuron_per_layer)), dim=0)
                    
                    kernel_matrix = torch.matmul(self.G.t(), self.G).to(dtype=self.dtype).to(device)
                    # self.U = torch.inverse(
                    #     torch.diag(torch.ones(self.G.shape[1]).to(dtype=self.dtype).to(device) * self.alg_lambda) + kernel_matrix).to(dtype=self.dtype).to(device)
                    
                elif self.alternative:
                    print('Computing Mahalanobis without online updates')
                    if self.initgrads_on_fly:
                        g_to_add = self.init_grads_on_demand(idx).to(self.device).to(dtype=self.dtype)
                    else:
                        g_to_add = self.init_grad_list[idx].to(self.device).to(dtype=self.dtype)
                    #raise NotImplementedError
                    if self.G is None:
                        self.G = (g_to_add.reshape(1, -1).to(self.device) / np.sqrt(self.neuron_per_layer)).to(dtype=self.dtype)
                    else:
                        self.G = torch.cat((self.G, g_to_add.reshape(1, -1).to(self.device) / np.sqrt(self.neuron_per_layer)), dim=0).to(dtype=self.dtype)
                    
                    kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype).to(self.device)
                    # self.U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                    #                        + kernel_matrix).to(dtype=self.dtype)


                else:
                    #print('Using Diagonal Approx. to Confidences')
                    if self.initgrads_on_fly:
                        g_to_add = self.init_grads_on_demand(idx).to(self.device).to(dtype=self.dtype)
                    else:
                        g_to_add = self.init_grad_list[idx].to(self.device).to(dtype=self.dtype)
                    self.U += (g_to_add * g_to_add / self.neuron_per_layer).to(self.device)  # U is diagonal, so represent as a vector
                    #containing only the diagonal elements and carry out computatuions accordingly, i.e. GG^T becomes hadamaard(g,g), note
                    #that g vectors here are  computed for previously seen data as loop goes over (zip(indices,rewards))' i.e. G1,...,Gt
                    #print('U:', self.U)

        ######################################NOW, DO THE NECESSARY INVERSION ONCE THE G MATRICES HAVE BEEN FULLY UPDATES WITH ALL NEW INDICES#######################
        if self.alternative:
            self.U = (torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                                           + kernel_matrix).to(dtype=self.dtype)).to(self.device_select)
        if self.complete_cov_mat:
            self.U = torch.inverse(
                        torch.diag(torch.ones(self.G.shape[1]).to(dtype=self.dtype).to(device) * self.alg_lambda) + kernel_matrix).to(dtype=self.dtype).to(device)
            
    
    def add_data_prll(self, indices, rewards):
        # add the new observations, only if it didn't exist already
        #print("Shape:", np.array(self.init_grad_list).shape)
        for idx, reward in zip(indices, rewards):
            #if idx not in self.data['graph_indices']: #TODO: uncomment?
            self.data['graph_indices'].append(idx)
            self.data['rewards'].append(reward)

            self.collected_indices.append(idx)

            '''
            ALSO, KEEP TRACK OF THE UNIQUELY COLLECTED PTS AND THEIR WEIGHTS WHICH ARE INVERSELY PROPORTIONAL TO THE 
            VALUE OF THE STEP IN WHICH THEY WERE ACQUIRED FIRST
            '''
            if idx not in self.unique_data['graph_indices']:
                self.unique_data['graph_indices'].append(idx)
                #self.unique_data['rewards'].append(reward)
                #self.unique_data['weights'] = np.flip(np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices']))))
                #self.unique_data['weights'] = np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices'])))
            
        if self.no_var_computation:
            print('Post Vars will not be computed')
            pass
        else:
            if self.online:
                print("Using Online Mahalanobis")
                raise NotImplementedError()
            elif self.complete_cov_mat:
                print("Using complete cov matrix")
                raise NotImplementedError()
            elif self.alternative:
                print('Computing Mahalanobis without online updates')
                if self.initgrads_on_fly:
                    g_to_add = self.init_grads_on_demand(indices).to(self.device).to(dtype=self.dtype)
                else:
                    g_to_add = torch.stack(self.init_grad_list)[indices].to(self.device).to(dtype=self.dtype)
                #raise NotImplementedError
                if self.G is None:
                    self.G = (g_to_add.reshape(len(indices), -1).to(self.device) / np.sqrt(self.neuron_per_layer)).to(dtype=self.dtype)
                else:
                    self.G = torch.cat((self.G, g_to_add.to(self.device) / np.sqrt(self.neuron_per_layer)), dim=0).to(dtype=self.dtype)
                
                kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype).to(self.device)
                # self.U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                #                        + kernel_matrix).to(dtype=self.dtype)
            else:
                print('Using Diagonal Approx. to Confidences')
                raise NotImplementedError()
    ######################################NOW, DO THE NECESSARY INVERSION ONCE THE G MATRICES HAVE BEEN FULLY UPDATES WITH ALL NEW INDICES#######################
            if self.alternative:
                self.U = (torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                                                + kernel_matrix).to(dtype=self.dtype)).to(self.device_select)
            if self.complete_cov_mat:
                self.U = torch.inverse(
                            torch.diag(torch.ones(self.G.shape[1]).to(dtype=self.dtype).to(device) * self.alg_lambda) + kernel_matrix).to(dtype=self.dtype).to(device)
            

    def select(self, dummy = None): #COMPUTE POSTERIOR MEAN AND VARIANCES FOR ALL POSSIBLE CANDIDATES IN ACTION SET (WHERE SELF.U IS COMPUTED FROM PTS ADDEDF SO FAR), AND SELECTT CANDIDATE BASED ON UCB
        #print("Applying UCB Based Selection")
        ucbs = []

        if self.no_var_computation:
            print('Selecting Using No var computation')
            if self.pool:
                if self.ucb_wo_replacement:
                    unseen_indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                    if self.pool_top_means:
                        unseen_indices_means = self.data['means'][unseen_indices]
                        indices = unseen_indices[np.argpartition(unseen_indices_means, -self.pool_num)[-self.pool_num:]]
                    else:
                        indices = self._rds.choice(unseen_indices, self.pool_num, replace=False)
                    #print('Unseen pts:', unseen_indices.shape)
                    #print('Collected pts:', len(self.unique_data['ucb_replacement_graph_indices']))
                else:
                    if self.pool_top_means:
                        indices = unseen_indices[np.argpartition(self.data['means'], -self.pool_num)[-self.pool_num:]]
                    else:
                        indices = self._rds.choice(range(self.num_actions), self.pool_num, replace=False)  
            else:
                if self.ucb_wo_replacement:
                    indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                else:
                    indices = np.arange(self.num_actions)
            post_vars = torch.zeros(len(indices)).to(self.device_select)
            #post_means = torch.tensor([self.get_post_mean(i) for i in range(len(indices))])
            post_means = torch.squeeze(torch.tensor([self.get_post_mean_print_every(indices)]))
            #print('Post vars shape:', post_vars.shape)
            #print('Post means shape:', post_means.shape)
            if self.bernoulli_selection:
                ber_param = 1/(len(self.data['graph_indices'])-self.num_pretrain_steps+1)
                coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                if coin_toss_result:
                    ucbs = post_means.to(self.device_select)
                else:
                    ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars
            else:
                ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars

            if self.select_K_together:
                ix_pool = torch.topk(ucbs, self.select_K).indices
            else:
                ix_pool = torch.argmax(ucbs).item()
            if self.pool:
                ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
            else:
                ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]

        elif self.oracle:
            print('Selecting Using the Oracle')
            if self.pool:
                if self.ucb_wo_replacement:
                    unseen_indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                    if self.pool_top_means:
                        unseen_indices_means = self.data['means'][unseen_indices]
                        indices = unseen_indices[np.argpartition(unseen_indices_means, -self.pool_num)[-self.pool_num:]]
                    else:
                        indices = self._rds.choice(unseen_indices, self.pool_num, replace=False)
                    #print('Unseen pts:', unseen_indices.shape)
                    #print('Collected pts:', len(self.unique_data['ucb_replacement_graph_indices']))
                else:
                    if self.pool_top_means:
                        indices = unseen_indices[np.argpartition(self.data['means'], -self.pool_num)[-self.pool_num:]]
                    else:
                        indices = self._rds.choice(range(self.num_actions), self.pool_num, replace=False)  
            else:
                if self.ucb_wo_replacement:
                    indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                else:
                    indices = np.arange(self.num_actions)
            post_vars = torch.zeros(len(indices)).to(self.device_select)
            #post_means = torch.tensor([self.get_post_mean(i) for i in range(len(indices))])
            #post_means = torch.tensor(np.array(self.QM9_Dataset.data.y)[indices,self.reward]).to(self.device)
            post_means = torch.tensor(np.array([self.QM9_Dataset[i].y.reshape(-1,1)[0] for i in indices]))
            #print('Post vars shape:', post_vars.shape)
            #print('Post means shape:', post_means.shape)
            if self.bernoulli_selection:
                ber_param = 1/(len(self.data['graph_indices'])-self.num_pretrain_steps+1)
                coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                if coin_toss_result:
                    ucbs = post_means.to(self._select)
                else:
                    ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars
            else:
                ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars

            if self.select_K_together:
                ix_pool = torch.topk(ucbs, self.select_K).indices
            else:
                ix_pool = torch.argmax(ucbs).item()

            if self.pool:
                ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
            else:
                ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
    
        else:

            if self.online: #VECTORIZE COMPUTATIONS TO BE FASTER!!!
                print('Selecting Using Online Var computation')

                kx_t_matrix = (torch.matmul(torch.stack(self.init_grad_list), torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer) #num_actions x t matrix, each row is kxt of different action
                #print(torch.sum(torch.matmul(kx_t_matrix, self.Kt_inv) * kx_t_matrix, dim=1).shape)
                post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ) / self.neuron_per_layer - torch.sum(torch.matmul(kx_t_matrix, self.Kt_inv) * kx_t_matrix, dim=1))
                post_means = torch.tensor([self.get_post_mean(i) for i in range(self.num_actions)])
                ucbs = post_means + np.sqrt(self.exploration_coef) * post_vars
                ix = torch.argmax(ucbs).item()
                #print(ix)
                #print((self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ).shape))
                #print(torch.matmul(torch.matmul(kx_t_matrix, self.Kt_inv), kx_t_matrix.t()).shape)
                #print(post_vars.shape)

            elif self.alternative:
                if self.thompson_sampling:
                    print('Selecting Using Mahalanobis Based Thompson Sampling')
                else:
                    print('Selecting Using Mahalanobis Based Posterior Computation')

                if self.pool:
                    if self.ucb_wo_replacement:
                        unseen_indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                        if self.pool_top_means:
                            unseen_indices_means = self.data['means'][unseen_indices]
                            indices = unseen_indices[np.argpartition(unseen_indices_means, -self.pool_num)[-self.pool_num:]]
                        else:
                            indices = self._rds.choice(unseen_indices, self.pool_num, replace=False)
                        #print('Unseen pts:', unseen_indices.shape)
                        #print('Collected pts:', len(self.unique_data['ucb_replacement_graph_indices']))
                    else:
                        if self.pool_top_means:
                            indices = unseen_indices[np.argpartition(self.data['means'], -self.pool_num)[-self.pool_num:]]
                        else:
                            indices = self._rds.choice(range(self.num_actions), self.pool_num, replace=False)  
                else:
                    if self.ucb_wo_replacement:
                        indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                    else:
                        indices = np.arange(self.num_actions)

                #print('Size of pool after eliminating repetitions:', len(indices))

                #kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype).to(self.device)
                #self.U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                #                           + kernel_matrix).to(dtype=self.dtype)

                if self.initgrads_on_fly:
                    g_vectors = torch.stack(self.init_grads_on_demand(indices)).to(self.device_select).to(dtype=self.dtype)
                else:
                    g_vectors = torch.stack(self.init_grad_list_cpu)[indices].to(self.device_select).to(dtype=self.dtype)

                if self.initgrads_on_fly:
                    g_vectors_observed = torch.stack(self.init_grads_on_demand(self.data['graph_indices'])).to(self.device_select).to(dtype=self.dtype)
                else:
                    g_vectors_observed = torch.stack(self.init_grad_list_cpu)[self.data['graph_indices']].to(self.device_select).to(dtype=self.dtype)

                if len(self.data['graph_indices']) <= 1:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=self.dtype) / self.neuron_per_layer)
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g_vectors * g_vectors_observed, dim=1 ) / self.neuron_per_layer - torch.sum(kx_t_matrix * self.U.to(self.device_select) * kx_t_matrix, dim=1))
                else:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=self.dtype) / self.neuron_per_layer)
                    #print('kxt:', kx_t_matrix.shape)
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    #print(kx_t.shape)
                    post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g_vectors * g_vectors, dim=1 ) / self.neuron_per_layer - torch.sum(torch.matmul(kx_t_matrix, self.U.to(self.device_select)) * kx_t_matrix, dim=1))
                    #print('post_var:', post_vars.shape)
                #post_means = torch.tensor([self.get_post_mean(indices[i]) for i in range(len(indices))])
                post_means = self.data['means'][indices]
                if self.bernoulli_selection:
                    ber_param = 1/(len(self.data['graph_indices'])-self.num_pretrain_steps+1)
                    coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                    if coin_toss_result:
                        ucbs = post_means.to(self.device_select)
                    else:
                        if self.thompson_sampling:
                            ucbs = torch.tensor([random.normal(loc=m, scale=s, size=1) for m,s in zip(post_means, np.sqrt(self.exploration_coef)*post_vars)])
                        else:
                            ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars
                else:
                    if self.thompson_sampling:
                        ucbs = torch.tensor([random.normal(loc=m, scale=s, size=1) for m,s in zip(post_means, np.sqrt(self.exploration_coef)*post_vars)])
                    else:
                        ucbs = post_means.to(self.device_select) + np.sqrt(self.exploration_coef) * post_vars

                if self.select_K_together:
                    ix_pool = torch.topk(ucbs, self.select_K).indices
                else:
                    ix_pool = torch.argmax(ucbs).item()

                if self.pool:
                    ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
                else:
                    ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
                #print('ix1:',ix)
                #print('ucbs1:', ucbs1)

            elif self.complete_cov_mat:
                print('Selecting Using Complete Cov Matrix')
                if self.G is None:
                    post_vars = torch.sqrt(torch.sum(self.exploration_coef * torch.sum(torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), dim=1).to(dtype=self.dtype) * self.U.to(dtype=self.dtype) / self.neuron_per_layer))
                else:
                    if self.U.shape[0] <= 1:
                        post_vars = torch.sqrt(torch.sum(self.exploration_coef * torch.sum(torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), dim=1).to(dtype=self.dtype) * self.U.to(dtype=self.dtype) / self.neuron_per_layer))
                    else:
                        post_vars = torch.sqrt(torch.sum(torch.matmul(torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), self.U.to(dtype=self.dtype)) * torch.stack(self.init_grad_list).to(device).to(dtype=self.dtype), dim=1))
                post_means = torch.tensor([self.get_post_mean(i) for i in range(self.num_actions)])
                ucbs = post_means + np.sqrt(self.exploration_coef) * post_vars
                ix = torch.argmax(ucbs).item()

            else:  
                print('selecting Using Diagonal Approx.')  
                if self.pool:
                    if self.ucb_wo_replacement:
                        unseen_indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                        if self.pool_top_means:
                            unseen_indices_means = self.data['means'][unseen_indices]
                            indices = unseen_indices[np.argpartition(unseen_indices_means, -self.pool_num)[-self.pool_num:]]
                        else:
                            indices = self._rds.choice(unseen_indices, self.pool_num, replace=False)
                        #print('Unseen pts:', unseen_indices.shape)
                        #print('Collected pts:', len(self.unique_data['ucb_replacement_graph_indices']))
                    else:
                        if self.pool_top_means:
                            indices = unseen_indices[np.argpartition(self.data['means'], -self.pool_num)[-self.pool_num:]]
                        else:
                            indices = self._rds.choice(range(self.num_actions), self.pool_num, replace=False)  
                else:
                    if self.ucb_wo_replacement:
                        indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                    else:
                        indices = np.arange(self.num_actions)

                
                post_means = torch.squeeze(torch.tensor([self.get_post_mean_print_every(indices)]))
                #Approx posterior mean by network output
                #g = torch.stack(self.init_grad_list)[indices].to(dtype=self.dtype) #dim_init_gradsxnum_indices
                if self.initgrads_on_fly:
                    g = self.init_grads_on_demand(indices).to(self.device).to(dtype=self.dtype)
                else:
                    g = torch.stack(self.init_grad_list)[indices].to(self.device).to(dtype=self.dtype)
                post_vars = (torch.sqrt(torch.sum( g * g / ( (self.U.to(self.device).to(dtype=self.dtype)/len(self.data['graph_indices']) + self.alg_lambda)),dim=1) / self.neuron_per_layer)).to(self.device) #eq8 in paper' just vectorized instead of matrix operations
                #due to diagonalization: since we approximate the inverse matrix in the middle (i.e. self.U) only by its diagonal elements'
                #in the operation g^TU^-1g we are simply reduced to the above operation
                #print('Post vars shape:', post_vars.shape)
                #print('Post means shape:', post_means.shape)

                if self.bernoulli_selection:
                    ber_param = 1/(len(self.data['graph_indices'])-self.num_pretrain_steps+1)
                    coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                    if coin_toss_result:
                        ucbs = post_means.to(self.device)
                    else:
                        ucbs = post_means.to(self.device) + np.sqrt(self.exploration_coef) * post_vars
                else:
                    ucbs = post_means.to(self.device) + np.sqrt(self.exploration_coef) * post_vars

                if self.select_K_together:
                    ix_pool = torch.topk(ucbs, self.select_K).indices.tolist()
                else:
                    ix_pool = torch.argmax(ucbs).item()
            
                if self.pool:
                    ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
                else:
                    ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]

        return ix

    def explore(self, dummy = None): #Exploration via pure random subsampling of actions, not even unc. sampling
        print('exploring')
        if self.select_K_together:
             ix = self._rds.choice(range(self.num_actions), size=self.select_K, replace = False).tolist()
        else:
             ix = self._rds.choice(range(self.num_actions))
        #ix = self._rds.choice(range(self.num_actions))
        return ix
    
    def run_exp(self):
        actions_prll = []
        print('RUN EXPLORE')
        pool = mp.Pool((mp.cpu_count()//2))
        for _ in range(self.batch_size):
            pool.apply_async(self.explore, args = (0,), callback = actions_prll.append, error_callback = handle_error)
        pool.close()
        pool.join()
        return actions_prll

    def run_sel(self):
        actions_prll = []
        print('RUN EXPLORE')
        pool = mp.Pool((mp.cpu_count()//2))
        for _ in range(self.batch_size):
            pool.apply_async(self.select, args = (0,), callback = actions_prll.append, error_callback = handle_error)
        pool.close()
        pool.join()
        return actions_prll

    def dummyy(self, dummy = None):
        print('dummy method')
        return self.dum.explore()
    
    def explore_throw_in(self): #Explore randomly unseen pts or via max of upper confıdence bounr, i.e. variance alone
        ix = self._rds.choice(list(set(range(self.num_actions)) - set(self.unique_data['graph_indices'])))
        print('Throwing in randomly explored point')
        #ix = self.select_unseen_variance()
        return ix

    def exploit(self): #Natuirally exploit only if aat least few previously explored points are
        #addded indeed to the prevously seen data, then, pick the candidate with highest reward seen so far
        if len(self.data['rewards'])>0:
            list_ix = np.argmax(self.data['rewards'])
            ix = self.data['graph_indices'][list_ix]
            return ix
        else:
            #return self.explore()
            return self._rds.choice(range(self.num_actions))

    def best_predicted(self): #Taking mean as a direct estimate of best predicted reward so far,
        #exploititively returns the best in action domain
        means = []
        for ix in range(self.num_actions):
            post_mean = self.func(self.action_domain[ix])
            means.append(post_mean.item())
        ix = np.argmax(means)
        return ix

    def get_post_var(self, idx): #Compute variance for any graph via the NTK Kernel as
        #approximated by the initialization gradients; I THINK NOT USED EXCEPT IN RUN_GNNUCB&RUN_PE
        if self.no_var_computation:
            return torch.zeros(1).item()
        else:
            if self.initgrads_on_fly:
                g = self.init_grads_on_demand(idx).to(self.device).to(dtype=self.dtype)
            else:
                g = self.init_grad_list[idx].to(self.device).to(dtype=self.dtype)

            if self.initgrads_on_fly:
                g_vectors_observed = self.init_grads_on_demand(self.data['graph_indices']).to(self.device).to(dtype=self.dtype)
            else:
                g_vectors_observed = torch.stack(self.init_grad_list)[self.data['graph_indices']].to(self.device).to(dtype=self.dtype)
            #g = self.init_grad_list[idx].to(self.device)
            #print(g.shape)

            if self.online:
                if len(self.data['graph_indices']) <= 1:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.t()) / self.neuron_per_layer
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    return (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g) / self.neuron_per_layer - torch.sum(kx_t * kx_t) * self.Kt_inv).item()
                else:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.t()) / self.neuron_per_layer
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    #print(kx_t.shape)
                    post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.Kt_inv), kx_t)).item()
                    #print('Post var:', post_var)
                    return post_var
                
            if self.complete_cov_mat:
                #raise NotImplementedError
                if self.G is None:
                    post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g * self.U / self.neuron_per_layer)).item()
                else:
                    if self.U.shape[0] <= 1:
                        post_var = torch.sqrt(torch.sum(g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer) * self.U * g / np.sqrt(self.neuron_per_layer))).item()
                    else:
                        post_var = torch.sqrt(torch.matmul(torch.matmul(g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer), self.U), g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer))).item()
                return post_var
            
            if self.alternative:

                #kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype).to(self.device)
                #self.U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                #                           + kernel_matrix).to(dtype=self.dtype)
                
                if len(self.data['graph_indices']) <= 1:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = (torch.matmul(g, g_vectors_observed.to(self.device).t())).to(dtype=self.dtype) / self.neuron_per_layer
                    return (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - kx_t.t() * self.U.to(self.device) * kx_t).item()
                else:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.to(self.device).t()).to(dtype=self.dtype) / self.neuron_per_layer
                    #print(torch.stack(self.init_grad_list)[self.data['graph_indices']].t().shape)
                    #print(kx_t.shape)
                    post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.U.to(self.device)), kx_t)).item()
                    #print('Post var:', post_var)
                    return post_var
            else:
                #return torch.sqrt(torch.sum(g * g / self.U ) / self.neuron_per_layer).item()
                return (torch.sqrt(torch.sum( g * g / ( (self.U.to(self.device).to(dtype=self.dtype)/len(self.data['graph_indices']) + self.alg_lambda))) / self.neuron_per_layer)).item()

    def get_post_var_print_every(self, indices):
        #approximated by the initialization gradients; I THINK NOT USED EXCEPT IN RUN_GNNUCB&RUN_PE
        if self.no_var_computation:
            return torch.zeros(len(indices)).tolist()
        else:
            if self.initgrads_on_fly:
                g = self.init_grads_on_demand(indices).to(self.device).to(dtype=self.dtype)
            else:
                g = torch.stack(self.init_grad_list)[indices].to(self.device).to(dtype=self.dtype)
            
            if self.initgrads_on_fly:
                g_vectors_observed = self.init_grads_on_demand(self.data['graph_indices']).to(self.device).to(dtype=self.dtype)
            else:
                g_vectors_observed = torch.stack(self.init_grad_list)[self.data['graph_indices']].to(self.device).to(dtype=self.dtype)
            #print(g.shape)

            if self.online:
                if len(self.data['graph_indices']) <= 1:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.t()) / self.neuron_per_layer
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    return (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g) / self.neuron_per_layer - torch.sum(kx_t * kx_t) * self.Kt_inv).item()
                else:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.t()) / self.neuron_per_layer
                    #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                    #print(kx_t.shape)
                    post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.Kt_inv), kx_t)).item()
                    #print('Post var:', post_var)
                    return post_var
                
            if self.complete_cov_mat:
                #raise NotImplementedError
                if self.G is None:
                    post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g * self.U / self.neuron_per_layer)).item()
                else:
                    if self.U.shape[0] <= 1:
                        post_var = torch.sqrt(torch.sum(g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer) * self.U * g / np.sqrt(self.neuron_per_layer))).item()
                    else:
                        post_var = torch.sqrt(torch.matmul(torch.matmul(g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer), self.U), g.to(dtype=self.dtype) / np.sqrt(self.neuron_per_layer))).item()
                return post_var
            
            if self.alternative:

                #kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype).to(self.device)
                #self.U = torch.inverse(torch.diag(torch.ones(self.G.shape[0]).to(self.device) * self.alg_lambda) \
                #                           + kernel_matrix).to(dtype=self.dtype)
                
                if len(self.data['graph_indices']) <= 1:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = (torch.matmul(g, g_vectors_observed.to(self.device).t())).to(dtype=self.dtype) / self.neuron_per_layer
                    post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g, dim=1).to(dtype=self.dtype) / self.neuron_per_layer - torch.sum(kx_t * self.U.to(self.device) * kx_t, dim=1))
                    #print('Post_var:',post_var.shape)
                    return post_var
                else:
                    #print('Computing variances based on Mahalanobis Expression')
                    kx_t = torch.matmul(g, g_vectors_observed.to(self.device).t()).to(dtype=self.dtype) / self.neuron_per_layer
                    #print(torch.stack(self.init_grad_list)[self.data['graph_indices']].t().shape)
                    #print(kx_t.shape)
                    post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g, dim=1).to(dtype=self.dtype) / self.neuron_per_layer - torch.sum(torch.matmul(kx_t, self.U.to(self.device)) * kx_t, dim=1))
                    #print('Post var:', post_var)
                    #print('Post_var:',post_var.shape)
                    return post_var.cpu().tolist()
            
            
            else:
                #print('Getting post vars print every from diagonal approx')
                #g = torch.stack(self.init_grad_list)[indices].to(self.dtype) #dim_init_gradsxnum_indices
                #post_var = (torch.sqrt(torch.sum( g * g / self.U, dim=1) / self.neuron_per_layer))
                post_var = (torch.sqrt(torch.sum( g * g / ( (self.U.to(self.device).to(dtype=self.dtype)/len(self.data['graph_indices']) + self.alg_lambda)),dim=1) / self.neuron_per_layer))

                return post_var
            
    def compute_means_for_sel(self):

        self.data['means'] = torch.tensor(self.get_post_mean_print_every(range(self.num_actions)))


    def get_post_mean(self, idx): #I THINK ALSO NOT USED FURHTER HERE

        with torch.no_grad():

            if self.batch_GD:

                #ix, (features, rewards, edge_indices, batch) = list(enumerate(self.dataloader_init_grad))[idx]
                data = self.QM9_Dataset_init_grad[idx]
                #return self.func(data).item()
                return(self.func(data.to(self.device))).cpu().item()
                
            else:
                return self.func(self.action_domain[idx]).cpu().item()
    
    def get_post_mean_print_every(self, indices): #I THINK ALSO NOT USED FURHTER HERE

        with torch.no_grad():
            means = []
            if self.batch_GD:
                #ix, (features, rewards, edge_indices, batch) = list(enumerate(self.dataloader_init_grad))[idx]
                data = self.QM9_Dataset_init_grad[indices]
                loader = DataLoader(data, batch_size=100, shuffle=False)
                for i, d in enumerate(loader):
                    means.append(self.func(d.to(self.device)))
                means_flat = [item.item() for sublist in means for item in sublist]
                return means_flat

                #return self.func(data).item()
                #return(self.func(data.to(self.device))).item()
                
            else:
                return self.func(self.action_domain[indices]).item()
            
    def train_batch(self):
        print(f"Training in batches of {self.GD_batch_size} samples")
        if self.train_from_scratch:
            print('Train from scratch')
            self.func.load_state_dict(self.f0.state_dict())

        if self.focal_loss:
            focalloss=FocalLoss(device = self.device, beta=self.alpha,gamma=self.gamma)

        #self.func.train()

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr,) #weight_decay=self.alg_lambda)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=self.factor, patience=self.patience,
                                                       min_lr=0.000001)
        #index = list(np.arange(len(np.unique(self.unique_data['graph_indices'])))) #When training only over uniquely collected datapts

        if self.subsample:
            assert self.subsample_num <= len(self.data['graph_indices']), "Datapts to be subsampled is more than the number of non-unique data collected so far!"
            print(f"Subsampling {self.subsample_num} datapts for GD")
            if self.subsample_method == 'random':
                weight_arr = np.ones(len(self.data['graph_indices']))/len(self.data['graph_indices'])
                subset_indices = self._rds.choice(np.arange(len(self.data['graph_indices'])), size=self.subsample_num, p=weight_arr, replace=False).astype('int')
                dataloader = DataLoader(self.QM9_Dataset[subset_indices], batch_size=self.GD_batch_size, shuffle=True)
        elif self.batch_window:
            if self.batch_window_size >= len(self.data['graph_indices']):
                print('Number of collected samples has not reached batch_window_size yet, taking minimum of the two!!!')
            self.effective_batch_window_size = min(self.batch_window_size, len(self.data['graph_indices']))
            assert self.batch_window_size >= self.GD_batch_size
            print(f"Taking the last {self.effective_batch_window_size} datapts for GD")
            dataloader = DataLoader(self.QM9_Dataset[self.data['graph_indices'][-self.effective_batch_window_size:]], batch_size=self.GD_batch_size, shuffle=True)
        else:
            dataloader = DataLoader(self.QM9_Dataset[self.data['graph_indices']], batch_size=self.GD_batch_size, shuffle=True)

        length = len(dataloader)
        #print('length_dataloader:', length)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            #lr = scheduler.optimizer.param_groups[0]['lr']
            for i, data in enumerate(dataloader):

                #print('Features:', features)
                #print('Features_shape:', features.shape)
                #print('Rewards:', rewards)
                #print('Rewards_Shape:', rewards.shape)
                #print('Edge_Indices:', edge_indices)
                #print('edge_indices_shape:', edge_indices.shape)
                #print('Batch:', batch)
                #print('Batch_shape:', batch.shape)
                #print('Network_output:',self.func(features=features, batch=batch, edge_index=edge_indices))
                #print('Network_output_shape:', self.func(features=features, batch=batch, edge_index=edge_indices).shape)
                
                labels = data.y
                #print('Labels:', labels.shape)
                #deltas = self.func(data).to(device)- labels.to(device)
                #deltas = self.func(data, split=True).to(device)- labels.to(device)
                #label = self.unique_data['rewards'][ix]
                #delta = self.func(self.action_domain[self.unique_data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)

                optimizer.zero_grad()

                #print(deltas.shape)

                #losses = deltas * deltas #* self.unique_data['weights'][ix]

                if self.focal_loss:
                    #print('Using Focal Loss')
                    batch_loss = focalloss(inputs=self.func(data.to(self.device)), targets=labels.to(self.device))
                else:
                    batch_loss = torch.nn.functional.mse_loss(self.func(data.to(device)), labels.to(device))
                 
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += batch_loss.item()
                cnt += 1
                if cnt >= self.stop_count/self.GD_batch_size:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                        #print('The count is', cnt)
                    return tot_loss / cnt  #THESE RETURN STATEMENTSS BREAK THE WHILE STATEMENT AND THUS STOP GD
            #if len(epoch_losses) % 2 == 0:
                #val_error = self.test(self.dataloader_val_dataset)
                #scheduler.step(val_error)
                #print(f'Epoch: {len(epoch_losses):03d}, LR: {lr:7f}, SELF_LR: {self.lr:7f}, 'f'Val MAE: {val_error:.7f}')
            #delta2 = epoch_losses[-2]-epoch_losses[-1]
            #delta1 = epoch_losses[-3]-epoch_losses[-2]
            #print('delta1:',delta1)
            #print('delta2:',delta2)
            #relative_improvement = abs((delta1-delta2)/delta1)
            #relative_improvement = abs((delta1-delta2))
            #print("Delta Differences:", abs(delta1-delta2))
            #print('Threshold:', self.relative_improvement)
            #print('Loss', epoch_loss/length)
            #if relative_improvement < self.relative_improvement:
                #if self.verbose: #THE CRITERIONS IN THE PAPER!!!!
                    #print('Loss curve is getting flat, and the count is', cnt)
                    #print('The loss is', epoch_loss / length)
                    #print("Relative Improvement:", relative_improvement)
                #return epoch_loss / length
            if epoch_loss / length <= self.small_loss:  # stop training if the average loss is less than 0.0001
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length
            
    def test(self, loader):
        self.func.eval()
        error = 0

        for data in loader:
            data = data.to(self.device)
            #error += (self.func(data) * self.std - data.y * self.std).abs().sum().item()  # MAE
            error += (self.func(data) * self.std - data.y * self.std).abs().sum().item()
        return error / len(loader.dataset)


class PhasedGnnUCB(GnnUCB):
    def __init__(self, net: str,num_nodes: int, feat_dim: int, num_actions: int, action_domain: list,
                 alg_lambda: float = 1, exploration_coef: float = 1, t_intersect: int= np.inf,
                 num_mlp_layers: int = 2, neuron_per_layer: int = 128, lr: float = 1e-3,
                 nn_aggr_feat = True, train_from_scratch = False, verbose = True,
                 nn_init_lazy: bool = True, complete_cov_mat: bool = False, random_state = None, path: Optional[str] = None, **kwargs):
        super().__init__(net = net, num_nodes = num_nodes, feat_dim = feat_dim, num_actions = num_actions, action_domain = action_domain,
        alg_lambda = alg_lambda, exploration_coef = exploration_coef,
        num_mlp_layers = num_mlp_layers, neuron_per_layer = neuron_per_layer, lr = lr,
        nn_aggr_feat = nn_aggr_feat, train_from_scratch = train_from_scratch, verbose = verbose,
        nn_init_lazy = nn_init_lazy, complete_cov_mat = complete_cov_mat, random_state = random_state, path = path)
        self.maximizers = [i for i in range(self.num_actions)]
        self.t_intersect = t_intersect
        if net == 'NN':
            self.name = 'PhasedNN-UCB'
        else:
            self.name = 'PhasedGNN-UCB'

    def select(self):
        ucbs = []
        lcbs = []
        vars = []
        for ix in range(self.num_actions):
            post_mean = self.func(self.action_domain[ix])
            g = self.init_grad_list[ix]
            post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer)
            vars.append(post_var.item())
            ucbs.append(post_mean.item() + np.sqrt(self.exploration_coef ) * post_var.item())
            lcbs.append(post_mean.item() - np.sqrt(self.exploration_coef ) * post_var.item())
            # ucbs.append(post_mean.item() + np.sqrt(self.exploration_coef * ) * post_var.item())
            # lcbs.append(post_mean.item() - np.sqrt(self.exploration_coef * ) * post_var.item())
        #max_lcb = np.max(lcbs)
        t = len(self.data['graph_indices'])
        if t > self.t_intersect:
            max_lcb = np.max([lcbs[i] for i in self.maximizers])
            self.maximizers = [i for i in self.maximizers if max_lcb <= ucbs[i]]
            print('intersecting...')
        else:
            max_lcb = np.max(lcbs)
            self.maximizers = [i for i in range(len(ucbs)) if max_lcb <= ucbs[i]]
        maximizer_vars = [vars[i] for i in self.maximizers]
        ix = self.maximizers[np.argmax(maximizer_vars)]
        return ix

    def train(self):
        if self.train_from_scratch:
            self.func.load_state_dict(self.f0.state_dict())

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)#, weight_decay=self.alg_lambda)

        index = list(np.arange(len(self.data['graph_indices'])))
        length = len(index)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            for ix in index:
                label = self.data['rewards'][ix]
                optimizer.zero_grad()
                delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                loss = delta * delta
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                    return tot_loss / cnt
            delta2 = epoch_losses[-2]-epoch_losses[-1]
            delta1 = epoch_losses[-3]-epoch_losses[-2]
            relative_improvement = (delta1-delta2)/delta1
            #if relative_improvement < 0.001:
            if relative_improvement < 0.001:
                if self.verbose:
                    print('Loss curve is getting flat, and the count is', cnt)
                    print('The loss is', epoch_loss / length)
                return epoch_loss / length
            #if epoch_loss / length <= 1e-4:  # stop training if the average loss is less than 0.0001

            if epoch_loss / length <= 1e-4:   
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length

