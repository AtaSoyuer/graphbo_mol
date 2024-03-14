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
from sklearn.preprocessing import Normalizer
import gpytorch
#from botorch.fit import fit_gpytorch_model

"""EXPLORATION COEF: BETA
LAMBDA: MSE REGULARIZATION, ALSO THE ALEOTORIC NOISE
COMPELTE COV MAT: COMPUTE POSTERIOR MATRICERS FULLY OR APPROX BY DIAGONALS
RANDOM STATE: RANDOM SEED
TRAIN FROM SCRATCH: ?"""

def handle_error(error):
    print(error, flush = True)

MAX_NUM_NODES = 29

def feat_pad(feat_mat):
    return torch.nn.functional.pad(feat_mat,pad=(0,0,0,MAX_NUM_NODES-len(feat_mat)), value=0)#value=float('nan'))
    
def z_pad(feat_mat):
    return torch.nn.functional.pad(feat_mat,pad=(0,MAX_NUM_NODES-len(feat_mat)), value=0)# value=float('nan'))  

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

class SS_alg:
    def __init__(self, qm9_data, qm9_val_data, init_grad_data, init_grad_loader, dataset_loader, val_dataset_loader, mean: float, std: float, num_actions:int,
                 feat_dim: int, num_nodes: int, dim: int, batch_GD: bool, exploration_coef: float = 1, lr: float = 1e-3, num_mlp_layers: int = 1, alg_lambda: float = 1,
                 action_domain: list = None, load_pretrained=False, subsample=False, subsample_method='random', verbose = True,  random_state = None, reward = 0, net = None,
                 subsample_num=20, pool=False, pool_num=20, large_scale=False, bernoulli_selection=False, ucb_wo_replacement=False, num_epochs = 100, thompson_sampling = False,
                 pool_top_means=False, select_K_together=False, select_K=5, batch_size = 50, pretrain_set = None, pretrain_labels = None, path: Optional[str] = None, **kwargs):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', self.device)
        print('Name:', torch.cuda.get_device_name(self.device))

        self.device_select = torch.device('cpu')
        
        self.GP_model = net
            
        self._rds = np.random if random_state is None else random_state
        self.alg_lambda = alg_lambda  # lambda regularization for the algorithm, FOR MSE LOSS
        self.exploration_coef = exploration_coef

        self.lr = lr

        self.load_pretrained = load_pretrained

        self.subsample = subsample
        self.subsample_method = subsample_method
        self.subsample_num = subsample_num

        self.verbose = verbose
    
        self.dtype = torch.float64

        self.QM9_Dataset = qm9_data
        self.QM9_Val_Dataset = qm9_val_data
        self.QM9_Dataset_init_grad = init_grad_data

        self.collected_indices = []

        self.dataloader_init_grad = init_grad_loader
        self.dataloader_dataset = dataset_loader
        self.dataloader_val_dataset = val_dataset_loader

        self.dim = dim

        self.mean = mean
        self.std = std

        self.num_actions = num_actions

        self.pool = pool
        self.pool_num  = pool_num

        self.large_scale=large_scale   

        self.bernoulli_selection = bernoulli_selection  

        self.ucb_wo_replacement = ucb_wo_replacement

        self.pool_top_means = pool_top_means

        self.reward = reward

        self.select_K_together = select_K_together
        self.select_K = select_K

        self.batch_size = batch_size

        self.num_epochs = num_epochs

        self.pretrain_set = pretrain_set
        self.pretrain_labels = pretrain_labels

        self.thompson_sampling = thompson_sampling

        if path is None: #BELOW JUST PUTS THE SPECIFIED PARAMS INSIDE THE CURLY BRACES IN RESPECTIVE ORDER
            self.path = 'trained_models/{}'.format('GP_RBF')
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

class SS_GP(SS_alg):  # Our main method
    # This class currently uses Woodbury's Identity(EQ8 IN PAPER). For scalability experiment, we need to use the regular gradient.
    def __init__(self, qm9_data, qm9_val_data, init_grad_data, init_grad_loader, dataset_loader, val_dataset_loader, mean: float, std: float, num_actions:int,
                 feat_dim: int, num_nodes: int, dim: int, batch_GD: bool, num_mlp_layers: int = 1, alg_lambda: float = 1, action_domain: list = None,
                 exploration_coef: float = 1, lr: float = 1e-3, random_state = None, reward = 0, net = None, pretrain_set = None, pretrain_labels = None,
                 load_pretrained=False, subsample=False, subsample_method='random', verbose = True, num_epochs = 100, thompson_sampling = False,
                 subsample_num=20, pool=False, pool_num=20, large_scale=False, bernoulli_selection=False, ucb_wo_replacement=False, 
                 pool_top_means=False, select_K_together=False, select_K=5, batch_size = 50, path: Optional[str] = None, **kwargs):
        super().__init__(qm9_data=qm9_data, qm9_val_data=qm9_val_data, init_grad_data=init_grad_data, init_grad_loader=init_grad_loader, dataset_loader=dataset_loader, num_epochs = num_epochs, num_actions=num_actions,
                         val_dataset_loader=val_dataset_loader, mean=mean, std=std, feat_dim=feat_dim, num_nodes=num_nodes, dim=dim, batch_GD=batch_GD, num_mlp_layers=num_mlp_layers, alg_lambda=alg_lambda,
                         exploration_coef = exploration_coef, lr=lr, random_state = random_state, reward = reward, net = net, load_pretrained=load_pretrained, subsample=subsample, subsample_method=subsample_method, 
                         verbose = verbose, subsample_num=subsample_num, pool=pool, pool_num=pool_num, large_scale=large_scale, bernoulli_selection=bernoulli_selection, ucb_wo_replacement=ucb_wo_replacement, thompson_sampling=thompson_sampling,
                         pool_top_means=pool_top_means, select_K_together=select_K_together, select_K=select_K, batch_size = batch_size, path=path, action_domain=action_domain, pretrain_set = pretrain_set, pretrain_labels = pretrain_labels, **kwargs)


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
        self.action_domain = action_domain

        self.x_train = []
        self.y_train = []

        self.x_val = []
        self.y_val = []

    def add_data_ucb_replacement(self, indices, rewards):
        for idx, reward in zip(indices, rewards):
            if idx not in self.unique_data['ucb_replacement_graph_indices']:
                self.unique_data['ucb_replacement_graph_indices'].append(idx)

    def create_train_val_data_matrices(self):

        for data in self.dataloader_dataset:
            self.y_train.extend([d.y for d in data])
            self.x_train.extend([torch.hstack(( feat_pad(d.x), feat_pad(d.pos), z_pad(d.z)[:,None] )).flatten() for d in data] )
        
        self.y_train = torch.tensor(self.y_train)
        self.x_train = torch.stack(self.x_train)

        self.transformer = Normalizer().fit(self.x_train)
        self.x_train = torch.tensor(self.transformer.transform(self.x_train))

        for data in self.dataloader_val_dataset:
            self.y_val.extend([d.y for d in data])
            self.x_val.extend([torch.hstack(( feat_pad(d.x), feat_pad(d.pos), z_pad(d.z)[:,None] )).flatten() for d in data] )
        
        self.y_val = torch.tensor(self.y_val)
        self.x_val = torch.tensor(self.transformer.transform(torch.stack(self.x_val))) if not len(self.x_val) == 0 else torch.tensor(self.x_val)

        print('y_train_shape:', self.y_train.shape)
        print('x_train_shape:', self.x_train.shape)

        print('y_val_shape:', self.y_val.shape)
        print('x_val_shape:', self.x_val.shape)
    
    def add_data_prll(self, indices, rewards):
        print(f'adding {len(indices)} many data')
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
                #self.unique_data['weights'] = np.reciprocal(np.linspace(1,len(self.unique_data['graph_indices']),num=len(self.unique_data['graph_indices']))

    
    def explore_throw_in(self): #Explore randomly unseen pts or via max of upper confıdence bounr, i.e. variance alone
        ix = self._rds.choice(list(set(range(self.num_actions)) - set(self.unique_data['graph_indices'])))
        print('Throwing in randomly explored point')
        #ix = self.select_unseen_variance()
        return ix

    def explore(self): #Exploration via pure random subsampling of actions, not even unc. sampling
        print('exploring')
        if self.select_K_together:
             ix = self._rds.choice(range(self.num_actions), size=self.select_K, replace = False).tolist()
        else:
             ix = self._rds.choice(range(self.num_actions))
        #ix = self._rds.choice(range(self.num_actions))
        return ix
    
    def select(self): #COMPUTE POSTERIOR MEAN AND VARIANCES FOR ALL POSSIBLE CANDIDATES IN ACTION SET (WHERE SELF.U IS COMPUTED FROM PTS ADDEDF SO FAR), AND SELECTT CANDIDATE BASED ON UCB
        print("Applying UCB Based Selection")
        ucbs = []
        print('Selection')

        if self.pool:
            if self.ucb_wo_replacement:
                unseen_indices = np.array(list(set(range(self.num_actions)) - set(self.unique_data['ucb_replacement_graph_indices'])))
                #print('learner_unique_data:', self.unique_data['ucb_replacement_graph_indices'])
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

        self.GP_model.eval()
        self.GP_model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                
            observed_pred = self.GP_model.likelihood(self.GP_model(self.x_train[indices,:].to(self.device)))
            lower, upper = observed_pred.confidence_region()
            #print('observed_preds:', observed_pred.shape)
            #print('upper_conf_bound:', upper.shape)

        post_vars = ((upper - lower)/2.0)**2
       
        post_means = self.data['means'][indices]
        #print('post_means_device:', post_means.device)
        #print('post_means:', post_means[0])
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
            #print('ix_pool_device:', ix_pool)

        if self.pool:
            ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
        else:
            ix = indices[ix_pool.cpu()].tolist() if self.select_K_together else indices[ix_pool]
        #print('ix1:',ix)
        #print('ucbs1:', ucbs1)

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
        
    def set_GP_model(self, X_train, y_train, likelihood):

        self.GP_model = GP_gpy(x_train = X_train, y_train = y_train, likelihood=likelihood).to(self.device)

    def get_post_var(self, idx): #Compute variance for any graph via the NTK Kernel as
        #approximated by the initialization gradients; I THINK NOT USED EXCEPT IN RUN_GNNUCB&RUN_PE
        
        self.GP_model.eval()
        self.GP_model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            
            if isinstance(idx, (int, np.integer)):
                pred_pts = self.x_train[idx,:].unsqueeze(0)
            else:
                pred_pts = self.x_train[idx,:]

            print('SHAPE:', pred_pts.shape)
       
            observed_pred = self.GP_model.likelihood(self.GP_model(pred_pts.to(self.device)))
            lower, upper = observed_pred.confidence_region()

        post_vars = ((upper - lower)/2.0)**2

        return post_vars.cpu().tolist()

    def get_post_var_print_every(self, indices):
        
        self.GP_model.eval()
        self.GP_model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                
            observed_pred = self.GP_model.likelihood(self.GP_model(self.x_train[indices,:].to(self.device)))
            lower, upper = observed_pred.confidence_region()

        post_vars = ((upper - lower)/2.0)**2

        return post_vars.cpu().tolist()
    
            
    def compute_means_for_sel(self):

        self.data['means'] = torch.tensor(self.get_post_mean_print_every(range(self.num_actions)))


    def get_post_mean(self, idx): #I THINK ALSO NOT USED FURHTER HERE

        self.GP_model.eval()
        self.GP_model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                
            observed_pred = self.GP_model.likelihood(self.GP_model(self.x_train[idx,:].to(self.device))).mean.cpu()

        return observed_pred
    
    def get_post_mean_print_every(self, indices): #I THINK ALSO NOT USED FURHTER HERE

        self.GP_model.eval()
        self.GP_model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            observed_pred = []
            chunk_size = 200
            range_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
            for idxs in range_chunks:
                temp = self.GP_model.likelihood(self.GP_model(self.x_train[idxs,:].to(self.device))).mean.cpu()
                observed_pred.extend(temp if isinstance(temp, list) else temp.cpu().tolist())
                
            #observed_pred = self.GP_model.likelihood(self.GP_model(self.x_train[indices,:].to(self.device))).mean.cpu()

        return torch.tensor(observed_pred)
            
    def train_batch(self):
        print(f"Training GP with {len(self.data['graph_indices'])+len(self.pretrain_set)} samples")
        #NOTE:THE BELOW LINE RE-INITIALIZES GP WITH NEW DATA AND TRAINS FULLY WHEREAS THE LINE BELOW RE-TRAINS IT SORT OF.
        self.set_GP_model(X_train=torch.cat( (self.pretrain_set.to(self.device), self.x_train[self.data['graph_indices'],:].to(self.device)) , dim=0), y_train=torch.cat( (self.pretrain_labels.to(self.device), self.y_train[self.data['graph_indices']].to(self.device))), likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device))

        self.GP_model.train()
        self.GP_model.likelihood.train()

        #self.GP_model.set_train_data(inputs = torch.cat( (self.pretrain_set.to(self.device), self.x_train[self.data['graph_indices'],:].to(self.device)) , dim=0), targets = torch.cat( (self.pretrain_labels.to(self.device), self.y_train[self.data['graph_indices']].to(self.device)) ), strict=False)

        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.GP_model.likelihood, self.GP_model)
        #fit_gpytorch_model(self.loss)
        num_epochs = self.num_epochs

        optimizer = torch.optim.Adam(self.GP_model.parameters(), lr=self.lr)

        for i in range(num_epochs):

            optimizer.zero_grad()
            output = self.GP_model(torch.cat( (self.pretrain_set.to(self.device), self.x_train[self.data['graph_indices'],:].to(self.device)) , dim=0))
            loss = -self.loss(output, torch.cat( (self.pretrain_labels.to(self.device), self.y_train[self.data['graph_indices']].to(self.device)) ))
            loss.backward()
            print('Iter %d - Loss: %.3f   lengthscale: %.3f  noise: %.3f' % (
            i + 1, loss.item(),
            self.GP_model.cov.base_kernel.lengthscale.item(),
            self.GP_model.likelihood.noise.item()))
            optimizer.step()

        return loss
        