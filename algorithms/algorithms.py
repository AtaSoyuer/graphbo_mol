import numpy as np
import torch.optim as optim
import copy
from nets import NN, GNN, normalize_init
from nets_batched import GNN as GNN_batched
from nets_batched import NN as NN_batched
from typing import Optional
from config import *
import sys 
import os
sys.path.append(os.path.abspath("./base_code/graph_BO"))
from dataset_class import QM9_GNNUCB_Dataset 
from torch_geometric.loader import DataLoader

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    #lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    #batch = zip(*batch)
    data = [ torch.Tensor(t['graph'])[None,:].to(device) for t in batch ]
    #data = torch.nn.utils.rnn.pad_sequence(data).permute((1,0,2)) #PERMUTE BECAUSE THEIS PADDING FUNCTION
    #PUTS BATCH DIMENSION IN 2ND DIM FOR SOME REASON!!!
    #print('Collate data shape:', data.shape)
    #data = torch.cat(data, dim=0)
    rewards = [ torch.Tensor(t['reward']).to(device) for t in batch ]
    ## compute mask
    ##mask = (batch != 0).to(device)
    return data, rewards

"""EXPLORATION COEF: BETA
LAMBDA: MSE REGULARIZATION, ALSO THE ALEOTORIC NOISE
COMPELTE COV MAT: COMPUTE POSTERIOR MATRICERS FULLY OR APPROX BY DIAGONALS
RANDOM STATE: RANDOM SEED
TRAIN FROM SCRATCH: ?"""
class UCBalg:
    def __init__(self, net: str, feat_dim: int, num_nodes: int, batch_GD: bool, num_mlp_layers: int = 1, alg_lambda: float = 1,
                 exploration_coef: float = 1, neuron_per_layer: int = 100,
                 complete_cov_mat: bool = False, lr: float = 1e-3,
                 random_state = None, nn_aggr_feat: bool = True, 
                 train_from_scratch=False, stop_count=1000, relative_improvement=1e-4, small_loss = 1e-3, 
                 load_pretrained=False, dropout=False, dropout_prob=0.2, subsample=False, subsample_method='random', verbose = True,
                 subsample_num=20, greedy=False, online=False, alternative = False, GD_batch_size=10, path: Optional[str] = None, **kwargs):
    
        if batch_GD:
            print('Using the nets from nets_batched module')
            if net == 'NN': #TO BE ABLE TO COMPARE AGAINST NAIVE NTK
                self.func = NN_batched(input_dim=feat_dim * num_nodes, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(device)
            elif net == 'GNN':
                self.func = GNN_batched(input_dim=feat_dim, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat, dropout=dropout, dropout_prob=dropout_prob).to(device)
            else:
                raise NotImplementedError
        else:
            if net == 'NN': #TO BE ABLE TO COMPARE AGAINST NAIVE NTK
                self.func = NN(input_dim=feat_dim * num_nodes, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat).to(device)
            elif net == 'GNN':
                self.func = GNN(input_dim=feat_dim, depth=num_mlp_layers, width=neuron_per_layer, aggr_feats=nn_aggr_feat, dropout=dropout, dropout_prob=dropout_prob).to(device)
            else:
                raise NotImplementedError
            
        self._rds = np.random if random_state is None else random_state
        self.alg_lambda = alg_lambda  # lambda regularization for the algorithm, FOR MSE LOSS
        self.num_net_params = sum(p.numel() for p in self.func.parameters() if p.requires_grad) #COUNT ALL MODEL PARAMS THAT REQUIRE GRAD, I.E. TRAINABLE
        self.U = alg_lambda * torch.ones((self.num_net_params,)).to(device)
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
        self.dtype = torch.float32

        self.QM9_Dataset = QM9_GNNUCB_Dataset()
        self.GD_batch_size = GD_batch_size

        self.batch_GD = batch_GD

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
    def __init__(self, net: str,num_nodes: int, feat_dim: int, num_actions: int, action_domain: list, batch_GD: bool,
                 alg_lambda: float = 1, exploration_coef: float = 1, t_intersect: int = np.inf,
                 num_mlp_layers: int = 2, neuron_per_layer: int = 128, lr: float = 1e-3,
                 nn_aggr_feat = True, train_from_scratch = False, stop_count=1000, relative_improvement=1e-4, small_loss = 1e-3, load_pretrained=False, 
                 dropout=False, dropout_prob=0.2, subsample=False, subsample_method='random', subsample_num=20, greedy=False, online=False, verbose = True,
                 nn_init_lazy: bool = True, complete_cov_mat: bool = False, alternative=False, GD_batch_size=10, random_state = None, path: Optional[str] = None, **kwargs):
        super().__init__(net=net, feat_dim=feat_dim, num_mlp_layers=num_mlp_layers, alg_lambda=alg_lambda, verbose = verbose, #INHERITS THE BASE UCB ALGO CLASS
                         lr = lr, complete_cov_mat = complete_cov_mat, nn_aggr_feat = nn_aggr_feat, train_from_scratch = train_from_scratch, 
                         stop_count=stop_count, relative_improvement=relative_improvement, small_loss=small_loss, load_pretrained=load_pretrained, num_nodes=num_nodes,
                         exploration_coef=exploration_coef, dropout_prob=dropout_prob, dropout=dropout, neuron_per_layer=neuron_per_layer, random_state=random_state, 
                         subsample=subsample, subsample_method=subsample_method, subsample_num=subsample_num, greedy=greedy, online=online, alternative=alternative, GD_batch_size=GD_batch_size, 
                         batch_GD=batch_GD, path=path, **kwargs)

        self.nn_aggr_feat = nn_aggr_feat

        # Create the network for computing gradients and subsequently variance.
        if self.load_pretrained:
            #self.func = torch.load('/local/bsoyuer/base_code/graph_BO/results/saved_models/reward1_5epochs')
            self.func.load_state_dict(torch.load('/local/bsoyuer/base_code/graph_BO/results/saved_models/reward1_5epochs.pt'))
            self.func.train()
            self.f0 = copy.deepcopy(self.func)
            #self.f0 = normalize_init(self.f0)
            print("Loaded Pretrained Model")
        else:
            self.f0 = copy.deepcopy(self.func) #SO THAT AS NETWORK FUNCTION IS UPDATED, INITIALIZATION ISNT!
            self.f0 = normalize_init(self.f0)

            if nn_init_lazy:
                self.func = normalize_init(self.func)
                self.f0 = copy.deepcopy(self.func)

        if net == 'NN':
            self.name = 'NN-UCB'
        else:
            self.name = 'GNN-UCB'

        self.data = {
            'graph_indices': [], #GRAPH DATA
            'rewards': [], #CONTINOUS REWARD
            'weights': []
        }

        self.unique_data = {   #ONLY ADD IF PT NOT OBSERVED BEFORE
            'graph_indices': [],
            'rewards': [],
            'weights': []
        }

        self.num_actions = num_actions #Actıon set sıze
        self.action_domain = action_domain #List, Value Inputted In Other Code?

        self.init_grad_list = []
        self.get_init_grads()

    def save_model(self):
        super().save_model()
        torch.save(self.f0, self.path + "/f0_model")

    def get_init_grads(self):
        post_mean0 = []
        if self.batch_GD:
            for graph in self.action_domain:
                self.f0.zero_grad() #Clear The gradients computed by bacxkwards for previous Graph in domain
                if self.nn_aggr_feat:
                    post_mean0.append(self.f0(torch.tensor(graph.feat_mat_aggr_normed()[None,:]))) #Algortihm uses GNN output(but not at init?) as mean estimate
                else:
                    post_mean0.append(self.f0(torch.tensor(graph.feat_mat_normed()[None,:])))
                post_mean0[-1].backward(retain_graph=True) #Compute gradients wrt the last forward pass,
                #since we do multiple backwards on the same computational graph (with gradients cleared at each step) iteratively,
                #so we dont want the implicit computattions in the networw pretaining to f0 to be freed!!!
                # Get the Variance.
                g = torch.cat([p.grad.flatten().detach() for p in self.f0.parameters()]) #Backward computes grads' .grad method accesses them
                #Flatten the gradients wrt each parameter anc concatenates them end to end to gett a full gradient vecotr, g_theta(Graph)
                self.init_grad_list.append(g)
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
            

    # def get_small_cov(self, g: np.ndarray):
    #     # Need to check square root. In any case, it is an issue of scaling - sweeping over beta properly would work.
    #     k_xx = g.dot(g)
    #     k_xy = torch.matmul(g.reshape(1, -1), self.G.T)
    #     k_xy = torch.matmul(k_xy, self.U_inv_small)
    #     k_xy = torch.matmul(k_xy, torch.matmul(self.G, g.reshape(-1, 1)))
    #     final_val = k_xx - k_xy
    #     return final_val

    def add_data(self, indices, rewards):
        # add the new observations, only if it didn't exist already
        #print("Shape:", np.array(self.init_grad_list).shape)
        num_new_indices = len(indices)
        for idx, reward in zip(indices, rewards):
            #if idx not in self.data['graph_indices']: #TODO: uncomment?
            self.data['graph_indices'].append(idx)
            self.data['rewards'].append(reward)

            self.QM9_Dataset.add(torch.tensor(self.action_domain[idx].feat_mat_aggr_normed()), torch.tensor(reward))

            #self.data['weights'] = np.reciprocal(np.linspace(1,len(self.data['graph_indices']),num=len(self.data['graph_indices'])))
            '''
            ALSO, KEEP TRACK OF THE UNIQUELY COLLECTED PTS AND THEIR WEIGHTS WHICH ARE INVERSELY PROPORTIONAL TO THE 
            VALUE OF THE STEP IN WHICH THEY WERE ACQUIRED FIRST
            '''
            if idx not in self.unique_data['graph_indices']:
                self.unique_data['graph_indices'].append(idx)
                #self.unique_data['rewards'].append(reward)
                #self.unique_data['weights'] = np.flip(np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices']))))
                #self.unique_data['weights'] = np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices'])))

            if self.online:

                print("Computing inverse covariance online")
                #print(len(self.data['graph_indices']))
                g_to_add = self.init_grad_list[idx]
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
            if self.complete_cov_mat:
                print("Using complete cov matrix")
                g_to_add = self.init_grad_list[idx]
                #raise NotImplementedError
                if self.G is None:
                    self.G = g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)
                else:
                    self.G = torch.cat((self.G, g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)), dim=0)
                
                kernel_matrix = torch.matmul(self.G.t(), self.G)
                self.U = torch.inverse(
                    torch.diag(torch.ones(self.G.shape[1]) * self.alg_lambda) + kernel_matrix)
                
            if self.alternative:
                print('Computing Mahalanobis without online updates')
                g_to_add = self.init_grad_list[idx].to(dtype=self.dtype)
                #raise NotImplementedError
                if self.G is None:
                    self.G = (g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)).to(dtype=self.dtype)
                else:
                    self.G = torch.cat((self.G, g_to_add.reshape(1, -1) / np.sqrt(self.neuron_per_layer)), dim=0).to(dtype=self.dtype)
                
                kernel_matrix = torch.matmul(self.G, self.G.t()).to(dtype=self.dtype)
                self.U = torch.inverse(
                    torch.diag(torch.ones(self.G.shape[0]) * self.alg_lambda) + kernel_matrix).to(dtype=self.dtype)


            else:
                self.U += self.init_grad_list[idx] * self.init_grad_list[idx] / self.neuron_per_layer  # U is diagonal, so represent as a vector
                #containing only the diagonal elements and carry out computatuions accordingly, i.e. GG^T becomes hadamaard(g,g), note
                #that g vectors here are  computed for previously seen data as loop goes over (zip(indices,rewards))' i.e. G1,...,Gt

    def select(self): #COMPUTE POSTERIOR MEAN AND VARIANCES FOR ALL POSSIBLE CANDIDATES IN ACTION SET (WHERE SELF.U IS COMPUTED FROM PTS ADDEDF SO FAR), AND SELECTT CANDIDATE BASED ON UCB
        ucbs = []
        if self.online: #VECTORIZE COMPUTATIONS TO BE FASTER!!!

            kx_t_matrix = (torch.matmul(torch.stack(self.init_grad_list), torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer) #num_actions x t matrix, each row is kxt of different action
            #print(torch.sum(torch.matmul(kx_t_matrix, self.Kt_inv) * kx_t_matrix, dim=1).shape)
            post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ) / self.neuron_per_layer - torch.sum(torch.matmul(kx_t_matrix, self.Kt_inv) * kx_t_matrix, dim=1))
            ix = torch.argmax(post_vars).item()
            #print(ix)
            #print((self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ).shape))
            #print(torch.matmul(torch.matmul(kx_t_matrix, self.Kt_inv), kx_t_matrix.t()).shape)
            #print(post_vars.shape)

        if self.alternative:

            if len(self.data['graph_indices']) <= 1:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t_matrix = (torch.matmul(torch.stack(self.init_grad_list), torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer)
                #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ) / self.neuron_per_layer - torch.sum(kx_t_matrix * self.U * kx_t_matrix, dim=1))
            else:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t_matrix = (torch.matmul(torch.stack(self.init_grad_list), torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer)
                #print('kxt:', kx_t_matrix.shape)
                #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                #print(kx_t.shape)
                post_vars = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( torch.stack(self.init_grad_list).to(dtype=self.dtype) * torch.stack(self.init_grad_list).to(dtype=self.dtype), dim=1 ) / self.neuron_per_layer - torch.sum(torch.matmul(kx_t_matrix, self.U) * kx_t_matrix, dim=1))
                #print('post_var:', post_vars.shape)
            post_means = torch.tensor([self.get_post_mean(i) for i in range(self.num_actions)])
            ucbs1 = post_means + np.sqrt(self.exploration_coef) * post_vars
            ix = torch.argmax(ucbs1).item()
            #print('ix1:',ix)
            #print('ucbs1:', ucbs1)

        else:    
        #if True:   
            for ix in range(self.num_actions): #Go through indices corresponding to each graph in action domain 
                #if self.batch_GD:
                    #if self.nn_aggr_feat:
                        #post_mean = self.func(torch.tensor(self.action_domain[ix].feat_mat_aggr_normed()))
                    #else:
                        #post_mean = self.func(torch.tensor(self.action_domain[ix].feat_mat_normed()))
                #else:
                    #post_mean = self.func(self.action_domain[ix])
                post_mean = self.get_post_mean(ix) #Should be doing the same thing as the commented code above
                #Approx posterior mean by network output
                g = self.init_grad_list[ix]
                if self.complete_cov_mat:
                    #raise NotImplementedError
                    if self.G is None:
                        post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g * self.U / self.neuron_per_layer))
                    else:
                        if self.U.shape[0] <= 1:
                            post_var = torch.sqrt(torch.sum(g / np.sqrt(self.neuron_per_layer) * self.U * g / np.sqrt(self.neuron_per_layer)))
                        else:
                            post_var =  torch.sqrt(torch.matmul(torch.matmul(g / np.sqrt(self.neuron_per_layer), self.U), g / np.sqrt(self.neuron_per_layer)))
                
                #if self.online:
                
                    #kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()) / self.neuron_per_layer
                    #kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer
                    #print(kx_t.shape)
                    #post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g.to(dtype=self.dtype) * g.to(dtype=self.dtype) ) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.Kt_inv), kx_t))
                
                # if self.alternative:
                #     if len(self.data['graph_indices']) <= 1:
                #         #print('Computing variances based on Mahalanobis Expression')
                #         kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer
                #         #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                #         post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - kx_t.t() * self.U * kx_t)
                #     else:
                #         #print('Computing variances based on Mahalanobis Expression')
                #         kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer
                #         #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                #         #print(kx_t.shape)
                #         post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.U), kx_t))
                else:
                    # Use Approximate Covariance.
                    post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer) #eq8 in paper' just vectorized instead of matrix operations
                    #due to diagonalization: since we approximate the inverse matrix in the middle (i.e. self.U) only by its diagonal elements'
                    #in the operation g^TU^-1g we are simply reduced to the above operation
                ucbs.append(post_mean + np.sqrt(self.exploration_coef) * post_var)
            ix = np.argmax(ucbs)
            #print('ix2:',ix)
            #print('ucbs2:', ucbs)

        return ix
    
    def select_pool(self, pool): #COMPUTE POSTERIOR MEAN AND VARIANCES FOR ALL POSSIBLE CANDIDATES IN ACTION SET (WHERE SELF.U IS COMPUTED FROM PTS ADDEDF SO FAR), AND SELECTT CANDIDATE BASED ON UCB
        ucbs_pool = []
        means_pool = []
        #print("Action domain:", len(self.action_domain))
        #print("Pool:", max(pool))
        for ix in pool: #Go through indices corresponding to each graph in action domain 
            post_mean = self.func(self.action_domain[ix]) #Approx posterior mean by network output
            g = self.init_grad_list[ix]
            if self.complete_cov_mat:
                raise NotImplementedError
                # if self.G is None:
                #     post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g / self.U / self.neuron_per_layer))
                # else:
                #     post_var = np.sqrt(self.exploration_coef) * torch.sqrt(
                #         self.get_small_cov(g / np.sqrt(self.neuron_per_layer)))
            else:
                # Use Approximate Covariance.
                post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer) #eq8 in paper' just vectorized instead of matrix operations
                #due to diagonalization: since we approximate the inverse matrix in the middle (i.e. self.U) only by its diagonal elements'
                #in the operation g^TU^-1g we are simply reduced to the above operation
            ucbs_pool.append(post_mean.item() + np.sqrt(self.exploration_coef) * post_var.item())
            means_pool.append(post_mean.item())
        if self.greedy:
            ix = np.argmax(means_pool)
        else:
            ix = np.argmax(ucbs_pool)
        return pool[ix] #NECESSARY SINCE POOL HAS THE GRAPH INDICES ALL SCRAMBLED DUE TO CHOICE() SO ARGMAX
    #DOESNT NECESSARILY CORRESPOND TO THE RIGHT INDEX!!
    
    # def select_unseen_variance(self): #COMPUTE POSTERIOR MEAN AND VARIANCES FROM SAMPLES ADDED SO FAR, AND SELECTT CANDIDATE BASED ON MAXIMUM VARIANCE SO FAR
    #     variances = []
    #     for ix in list(set(range(self.num_actions)) - set(self.unique_data['graph_indices'])): #Go through indices corresponding to each graph in action domain 
    #         post_mean = self.func(self.action_domain[ix]) #Approx posterior mean by network output
    #         g = self.init_grad_list[ix]
    #         if self.complete_cov_mat:
    #             raise NotImplementedError
    #             # if self.G is None:
    #             #     post_var = torch.sqrt(torch.sum(self.exploration_coef * g * g / self.U / self.neuron_per_layer))
    #             # else:
    #             #     post_var = np.sqrt(self.exploration_coef) * torch.sqrt(
    #             #         self.get_small_cov(g / np.sqrt(self.neuron_per_layer)))
    #         else:
    #             # Use Approximate Covariance.
    #             post_var = torch.sqrt(torch.sum( g * g / self.U) / self.neuron_per_layer) #eq8 in paper' just vectorized instead of matrix operations
    #             #due to diagonalization, note that g vectors here are computed via generic query points G in action domain as opposed to those used to compute U
    #         variances.append(post_var.item())
    #     ix = np.argmax(variances)
    #     return ix
    

    def explore(self): #Exploration via pure random subsampling of actions, not even unc. sampling
        ix = self._rds.choice(range(self.num_actions))
        return ix
    
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
            return self.explore()

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
        g = self.init_grad_list[idx]
        #print(g.shape)

        if self.online:
            if len(self.data['graph_indices']) <= 1:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()) / self.neuron_per_layer
                #kx_t = (torch.matmul(g, torch.stack(self.init_grad_list, dim = 1)[:,self.data['graph_indices']].flatten(start_dim = 1)) / self.neuron_per_layer)
                return (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g) / self.neuron_per_layer - torch.sum(kx_t * kx_t) * self.Kt_inv).item()
            else:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()) / self.neuron_per_layer
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
                    post_var = torch.sqrt(torch.sum(g / np.sqrt(self.neuron_per_layer) * self.U * g / np.sqrt(self.neuron_per_layer))).item()
                else:
                    post_var =  torch.sqrt(torch.matmul(torch.matmul(g / np.sqrt(self.neuron_per_layer), self.U), g / np.sqrt(self.neuron_per_layer))).item()
            return post_var
        
        if self.alternative:
            if len(self.data['graph_indices']) <= 1:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t = (torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t())).to(dtype=self.dtype) / self.neuron_per_layer
                return (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - kx_t.t() * self.U * kx_t).item()
            else:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t = torch.matmul(g, torch.stack(self.init_grad_list)[self.data['graph_indices']].t()).to(dtype=self.dtype) / self.neuron_per_layer
                #print(torch.stack(self.init_grad_list)[self.data['graph_indices']].t().shape)
                #print(kx_t.shape)
                post_var = (self.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g * g).to(dtype=self.dtype) / self.neuron_per_layer - torch.matmul(torch.matmul(kx_t.t(), self.U), kx_t)).item()
                #print('Post var:', post_var)
                return post_var
        else:
            return torch.sqrt(torch.sum(g * g / self.U ) / self.neuron_per_layer).item()


    def get_post_mean(self, idx): #I THINK ALSO NOT USED FURHTER HERE
        if self.batch_GD:
            if self.nn_aggr_feat:
                return self.func(torch.tensor(self.action_domain[idx].feat_mat_aggr_normed()[None,:])).item()
            else:
                return self.func(torch.tensor(self.action_domain[idx].feat_mat_normed()[None,:])).item()
        else:
            return self.func(self.action_domain[idx]).item()

    
    """
    PRETRAIN() IS BASICALLY THE SAME AS TRAIN() AT THIS POINT, WITHOUT THE LATEST MODIFICATIONS
    """
    def pretrain(self): #pre_train_data): #Paper: Pure exploration to mimic pretraining of GNN
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr, weight_decay=self.alg_lambda/100)
        #self.data['graph_indices'].extend(pre_train_data['graph_indices']) #The Inputted set of Pretraining data are added to overall data seen
        #self.data['rewards'].extend(pre_train_data['rewards'])

        #index = list(np.arange(len(self.data['graph_indices'])))
        index = list(np.arange(len(np.unique(self.unique_data['graph_indices']))))

        length = len(index)
        np.random.shuffle(index) #Randomize order of pretraining set ~ pure exploration by random subsampling
        cnt = 0
        tot_loss = 0
        while True: #Represents each epoch as soon as we went over all indices in pretrain set
            epoch_loss = 0
            for ix in index:
                #label = self.data['rewards'][ix]
                label = self.unique_data['rewards'][ix]
                optimizer.zero_grad()
                #delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                delta = self.func(self.action_domain[self.unique_data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                loss = delta * delta #* self.unique_data['weights'][ix]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 500:
                    print('Too many steps, stopping GD.')
                    print('The loss is', tot_loss / cnt)  # train each epoch for J \leq 1000, THESE CRITERIA ARE DIFFERENT FROM THOSE IN THE PAPER APPENDIX?
                    return tot_loss / 1000
            if epoch_loss / length <= 1e-2:  # stop training if the average loss is less than 0.001
                print('Loss is getting flat, stopping GD.')
                print('The loss is', tot_loss / cnt)
                return epoch_loss / length

    def train(self):
        #print(self.train_from_scratch)
        if self.train_from_scratch:
            print('Train from scratch')
            self.func.load_state_dict(self.f0.state_dict())

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr,) #weight_decay=self.alg_lambda)
        #optimizer = optim.SGD(self.func.parameters(), lr=self.lr, weight_decay=self.alg_lambda)

        # index = list(np.arange(len(self.data['graph_indices'])))
        # # cnt = 0
        # # tot_loss = 0
        # for __ in range(10):
        #     epoch_loss = 0
        #     self._rds.shuffle(index)
        #     for ix in index:
        #         label = self.data['rewards'][ix]
        #         optimizer.zero_grad()
        #         delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
        #         loss = delta * delta
        #         loss.backward()
        #         optimizer.step()
        #         epoch_loss += loss.item()
        #         # tot_loss += loss.item()
        #         # cnt += 1
        # # return tot_loss / len(index) / 10
        # #self.save_model()

        """
        RANDOMLY SUBSAMPLE THE SAMPLES TO BE USED IN GD AT EACH STEP
        """
        if self.subsample:
            assert self.subsample_num <= len(self.data['graph_indices']), "Datapts to be subsampled is more than the number of non-unique data collected so far!"
            print(f"Subsampling {self.subsample_num} datapts for GD")
            if self.subsample_method == 'random':
                weight_arr = np.ones(len(self.data['graph_indices']))/len(self.data['graph_indices'])
            elif self.subsample_method == 'weights':
                histogram = np.histogram(self.data['graph_indices'], bins = np.arange(0, np.max(self.data['graph_indices'])+2))
                ##print(histogram[0].shape)
                ##print(np.max(self.data['graph_indices']))
                weights = [(histogram[0][x]).astype('float64') if x > 0.0 else 0.0 for x in self.data['graph_indices']]
                weight_arr = weights/np.sum(weights)
            elif self.subsample_method == 'inverse_weights':
                histogram = np.histogram(self.data['graph_indices'], bins = np.arange(0, np.max(self.data['graph_indices'])+2))
                ##print(histogram[0].shape)
                ##print(np.max(self.data['graph_indices']))
                weights = [(histogram[0][x]).astype('float64')**(-1) if x > 0.0 else 0.0 for x in self.data['graph_indices']]
                weight_arr = weights/np.sum(weights)
            subset_indices = self._rds.choice(np.arange(len(self.data['graph_indices'])), size=self.subsample_num, p=weight_arr, replace=False).astype('int')
            print(subset_indices)
            graph_subset = np.array(self.data['graph_indices'])[subset_indices]
            reward_subset = np.array(self.data['rewards'])[subset_indices]
            index = list(np.arange(len(graph_subset)))
        else:
            index = list(np.arange(len(self.data['graph_indices']))) #Over whole dataset now

        #index = list(np.arange(len(np.unique(self.unique_data['graph_indices'])))) #When training only over uniquely collected datapts

        length = len(index)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            for ix in index:
                if self.subsample:
                    label = reward_subset[ix]
                    delta = self.func(self.action_domain[graph_subset[ix]]).to(device)- torch.tensor(label).to(device)
                else:
                    label = self.data['rewards'][ix]
                    delta = self.func(self.action_domain[self.data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)
                #label = self.unique_data['rewards'][ix]
                #delta = self.func(self.action_domain[self.unique_data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)

                optimizer.zero_grad()
            
                loss = delta * delta #* self.unique_data['weights'][ix]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += loss.item()
                cnt += 1
                if cnt >= self.stop_count:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                    return tot_loss / cnt  #THESE RETURN STATEMENTSS BREAK THE WHILE STATEMENT AND THUS STOP GD
            delta2 = epoch_losses[-2]-epoch_losses[-1]
            delta1 = epoch_losses[-3]-epoch_losses[-2]
            #print('delta1:',delta1)
            #print('delta2:',delta2)
            #relative_improvement = abs((delta1-delta2)/delta1)
            relative_improvement = abs((delta1-delta2))
            #print("Delta Differences:", abs(delta1-delta2))
            #print('Threshold:', self.relative_improvement)
            #print('Loss', epoch_loss/length)
            if relative_improvement < self.relative_improvement:
                if self.verbose: #THE CRITERIONS IN THE PAPER!!!!
                    print('Loss curve is getting flat, and the count is', cnt)
                    print('The loss is', epoch_loss / length)
                    print("Relative Improvement:", relative_improvement)
                return epoch_loss / length
            if epoch_loss / length <= self.small_loss:  # stop training if the average loss is less than 0.0001
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length
            
    def train_batch(self):
        print(f"Training in batches of {self.GD_batch_size} samples")
        if self.train_from_scratch:
            print('Train from scratch')
            self.func.load_state_dict(self.f0.state_dict())

        optimizer = optim.Adam(self.func.parameters(), lr=self.lr,) #weight_decay=self.alg_lambda)

        #index = list(np.arange(len(np.unique(self.unique_data['graph_indices'])))) #When training only over uniquely collected datapts

        dataloader = torch.utils.data.DataLoader(self.QM9_Dataset, batch_size=self.GD_batch_size, shuffle=True, collate_fn = collate_fn_padd)
        #dataloader = DataLoader(self.QM9_Dataset, batch_size=self.GD_batch_size, shuffle=True, num_workers=0)
        length = len(dataloader)
        cnt = 0
        tot_loss = 0
        epoch_losses = []
        while True:
            epoch_loss = 0
            for ix, (sample_batched, rewards) in enumerate(dataloader):
                batch_loss = 0
                #print(sample_batched)
                #print(rewards)
                for sample, reward in zip(sample_batched, rewards):
                    #print('Sampled_Batch:', sample_batched.shape)
                    #print('Network output:', self.func(sample_batched).shape)
                    
                    label = reward
                    delta = self.func(sample).to(device)- torch.tensor(label).to(device)
                    #label = self.unique_data['rewards'][ix]
                    #delta = self.func(self.action_domain[self.unique_data['graph_indices'][ix]]).to(device)- torch.tensor(label).to(device)

                    optimizer.zero_grad()

                    #print(deltas.shape)

                    loss = delta * delta #* self.unique_data['weights'][ix]
                    batch_loss += torch.mean(loss)
                batch_loss = batch_loss / self.GD_batch_size
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                epoch_losses.append(epoch_loss)
                tot_loss += batch_loss.item()
                cnt += self.GD_batch_size
                if cnt >= self.stop_count:  # train each epoch for J \leq 1000
                    if self.verbose:
                        print('Too many steps, stopping GD.')
                        print('The loss is', tot_loss / cnt)
                    return tot_loss / cnt  #THESE RETURN STATEMENTSS BREAK THE WHILE STATEMENT AND THUS STOP GD
            delta2 = epoch_losses[-2]-epoch_losses[-1]
            delta1 = epoch_losses[-3]-epoch_losses[-2]
            #print('delta1:',delta1)
            #print('delta2:',delta2)
            #relative_improvement = abs((delta1-delta2)/delta1)
            relative_improvement = abs((delta1-delta2))
            #print("Delta Differences:", abs(delta1-delta2))
            #print('Threshold:', self.relative_improvement)
            #print('Loss', epoch_loss/length)
            if relative_improvement < self.relative_improvement:
                if self.verbose: #THE CRITERIONS IN THE PAPER!!!!
                    print('Loss curve is getting flat, and the count is', cnt)
                    print('The loss is', epoch_loss / length)
                    print("Relative Improvement:", relative_improvement)
                return epoch_loss / length
            if epoch_loss / length <= self.small_loss:  # stop training if the average loss is less than 0.0001
                if self.verbose:
                    print('Loss is getting small and the count is', cnt)
                    print('The loss is', epoch_loss/length)
                return epoch_loss / length


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

