import os
import csv
import sys
import json
from algorithms_pyg_mp import GnnUCB, UCBalg
import time
import numpy as np
import torch
from plot_scripts.utils_plot import plt_regret, plt_cumulative_rew
from matplotlib import pyplot as plt
from utils_exp_w_edgeix import NumpyArrayEncoder
import matplotlib
import argparse
import umap 
#import umap.umap_ as umap
import subprocess
import warnings
from dataset_class_w_edgeix import MyTransform, Complete 
from torch_geometric.loader import DataLoader
#from sklearn.model_selection import train_test_split
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import QM9
import matplotlib.colors as cm
import imageio
matplotlib.use('agg')
sys.path.append(os.path.abspath("./plot_scripts/")) 
sys.path.append(os.path.abspath("/cluster/scratch/bsoyuer/")) 
import bundles
import scipy.stats as st
from scipy.stats.stats import pearsonr
import multiprocessing as mp
import itertools
from supervised_pretraining import SupervisedPretrain



def evaluate(idx_list: list, reward_list: list, noisy: bool, _rds , noise_var: float): #Create reward array from given list of indices and add aleotoric noise if need be
    #print('indices:',idx_list)
    rew = np.array([reward_list[idx] for idx in idx_list])
    #if noisy:
        #rew = rew + _rds.normal(0, noise_var, size=rew.shape)
    return list(rew)

def main(args):

    # def response(result):
    #     global actions_prll
    #     actions_prll.append(result)

    def handle_error(error):
        print(error, flush = True)
    
    args.T = args.T if not args.select_K_together else int(float(args.T) / float(args.select_K))
    args.T0 = args.T0 if not args.select_K_together else int(float(args.T0) / float(args.select_K))
    args.T1 = args.T1 if not args.select_K_together else int(float(args.T1) / float(args.select_K))
    args.T2 = args.T2 if not args.select_K_together else int(float(args.T2) / float(args.select_K))
    args.pretrain_steps = args.pretrain_steps if not args.select_K_together else int(float(args.pretrain_steps) / float(args.select_K))
    args.print_every = args.print_every if not args.select_K_together else int(float(args.print_every) / float(args.select_K))
    args.batch_size = args.batch_size if not args.select_K_together else int(float(args.batch_size) / float(args.select_K))

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))

    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "Helvetica"
    # })

    print('Args:',args)
    
    #print('Run setting:', args.__dict__)

    # read full data
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

    if args.remove_smiles_for_sober:
        indices_to_remove_file = open("../SOBER/indices_of_smiles_toremove_github.csv", "r")
        indices_to_remove = list(csv.reader(indices_to_remove_file, delimiter=","))[0]
        indices_to_remove_file.close()
        indices_to_remove = [int(item) for item in indices_to_remove]
        print('INDICES TO REMOVE:', indices_to_remove)
        pretrain_dataset = []
        # idx = 0
        # for d in dataset:
        #     if idx not in indices_to_remove:
        #         pretrain_dataset.append(d)
        #     idx += 1
        # dataset = pretrain_dataset
        all_indices = list(np.arange(len(dataset)))
        remaining_indices = [idx for idx in all_indices if idx not in indices_to_remove]
        dataset = dataset[remaining_indices]
        print(len(dataset))
        args.num_actions -= len(indices_to_remove)
        args.dataset_size -= len(indices_to_remove)


    if args.load_pretrained:
        pretraining_alg = SupervisedPretrain(dim = args.dim, input_dim = args.feat_dim, width = args.neuron_per_layer, reward_plot_dir = args.reward_plot_dir, \
                                             pretrain_indices_name=args.pretrain_indices_name, model_name=args.pretrain_model_name, num_indices=args.pretrain_num_indices, \
                                             pretraining_load_pretrained=args.pretraining_load_pretrained, pretraining_pretrain_model_name=args.pretraining_pretrain_model_name, \
                                             laplacian_features=args.laplacian_features, laplacian_k=args.laplacian_k, std = std, dataset=dataset)
        pretraining_alg.train_loop()
        pretraining_alg.print_and_plot()

    # Split datasets.
    if args.load_pretrained:
        
        pretraining_indices = np.load(f'/cluster/scratch/bsoyuer/base_code/graph_BO/{args.pretrain_indices_name}.npy')

        #First mask out the indices used for pretraining
        if args.remove_pretrain:
            mask=np.full(len(dataset),True,dtype=bool)
            mask[pretraining_indices]=False
            dataset_without_pretraining=dataset[mask]
        else:
            mask=np.full(len(dataset),True,dtype=bool)
            dataset_without_pretraining=dataset[mask]

        print('Dataset w/o pretraining:', len(dataset_without_pretraining))

        if args.num_actions == args.dataset_size:  #TO PREVENT SHUFFLED INDICES WHE  USING WHOLE DATASET
            subset_indices = np.arange(args.dataset_size)
        else:
            subset_indices = np.random.choice(len(dataset_without_pretraining), args.num_actions, replace=False)

        mask=np.full(len(dataset_without_pretraining),True,dtype=bool)
        mask[subset_indices]=False
        dataset_removed=dataset_without_pretraining[mask]
        print('Dataset removed:', len(dataset_removed))

        #val_subset_indices = np.random.choice(len(dataset_removed), int(args.num_actions))
        val_subset_indices = np.random.choice(len(dataset_removed), 0, replace=False)

        #dataset_new = dataset_new.shuffle()
        train_dataset = dataset_without_pretraining[subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES
        val_dataset = dataset_removed[val_subset_indices]
        #train_dataset, val_dataset = train_test_split(dataset_subset, test_size=0.2, random_state=42)

        val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=25, shuffle=False)
        init_grad_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        print('Len Train Dataset:', len(train_dataset))
    
    else:
        subset_indices = np.random.choice(len(dataset), args.num_actions, replace=False)

        mask=np.full(len(dataset),True,dtype=bool)
        mask[subset_indices]=False
        dataset_removed=dataset[mask]
        print(len(dataset_removed))

        val_subset_indices = np.random.choice(len(dataset_removed), int(args.num_actions), replace=False)

        #dataset_new = dataset_new.shuffle()
        train_dataset = dataset_without_pretraining[subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES
        val_dataset = dataset_removed[val_subset_indices]
        #train_dataset, val_dataset = train_test_split(dataset_subset, test_size=0.2, random_state=42)

        val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=25, shuffle=False)
        init_grad_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        print(len(train_dataset))
    '''
    THIS HERE LIMITS THE READ DATASET TO NUM_ACTIONS MANY GRAPHS INTOTAL
    '''

    graph_data = [d for d in train_dataset]
    graph_rewards = [d.y for d in train_dataset]
    #print(graph_data)
    #print(graph_rewards)


    # if args.remove_smiles_for_sober:
    #     idx_search = np.argwhere(np.logical_and(15.7708 < np.array(graph_rewards)*std+mean, np.array(graph_rewards)*std+mean <= 15.7709))[0,0]
    # else:
    #     idx_search = 94761
    # print('idx_search:', idx_search)
    # print('15.7709 Value:', graph_rewards[idx_search]*std+mean)
    # sys.exit("Error message")

    max_reward = np.max(graph_rewards)
    max_reward_index = np.argmax(graph_rewards) #The maximum that we "dont know of"
    # set bandit algorithm
    assert args.num_actions == len(graph_data)
    assert len(graph_data) == len(graph_rewards)
    algo_rds = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    learner = GnnUCB(qm9_data=train_dataset, qm9_val_data=val_dataset, init_grad_data=train_dataset, init_grad_loader=init_grad_loader, dataset_loader=train_loader, val_dataset_loader=val_loader, 
                     net = args.net, feat_dim = args.feat_dim, dim = args.dim, num_nodes = args.num_nodes,num_actions = args.num_actions, action_domain = graph_data, verbose=args.runner_verbose,
                     alg_lambda = args.alg_lambda, exploration_coef = args.exploration_coef, train_from_scratch=args.train_from_scratch, nn_aggr_feat=args.nn_aggr_feat, no_var_computation=args.no_var_computation,
                     num_mlp_layers = args.num_mlp_layers_alg, neuron_per_layer = args.neuron_per_layer, lr = args.lr, nn_init_lazy=args.nn_init_lazy, stop_count=args.stop_count, reward = args.reward,
                     relative_improvement=args.relative_improvement, small_loss=args.small_loss, load_pretrained=args.load_pretrained, pretrain_model_name=args.pretrain_model_name, explore_threshold=args.explore_threshold, 
                     dropout=args.dropout, dropout_prob=args.dropout_prob, subsample=args.subsample, subsample_method=args.subsample_method, subsample_num=args.subsample_num, greedy=args.greedy, 
                     online=args.online_cov, complete_cov_mat=args.complete_cov_mat, alternative=args.alternative, GD_batch_size = args.GD_batch_size, mean=mean, std=std, factor = args.factor, patience = args.patience, 
                     batch_GD = args.batch_GD, pool = args.pool, pool_num = args.pool_num, batch_window=args.batch_window, batch_window_size=args.batch_window_size, focal_loss=args.focal_loss, alpha=args.alpha, gamma=args.gamma, 
                     large_scale=args.large_scale, bernoulli_selection=args.bernoulli_selection, ucb_wo_replacement = args.ucb_wo_replacement, small_net_var = args.small_net_var, initgrads_on_fly = args.initgrads_on_fly, oracle=args.oracle, 
                     select_K_together=args.select_K_together, select_K = args.select_K, batch_size = args.batch_size, laplacian_k = args.laplacian_k, thompson_sampling = args.thompson_sampling, random_state=algo_rds)

    #Initialize the algortihm & the model as above with learner 

    def run_explore(dummy = None): #Exploration via pure random subsampling of actions, not even unc. sampling
        #print('exploring')
        if learner.select_K_together:
             ix = learner._rds.choice(range(learner.num_actions), size=learner.select_K, replace = False).tolist()
        else:
             ix = learner._rds.choice(range(learner.num_actions))
        #ix = learner._rds.choice(range(learner.num_actions))
        return ix
    
    def run_select(dummy = None): #COMPUTE POSTERIOR MEAN AND VARIANCES FOR ALL POSSIBLE CANDIDATES IN ACTION SET (WHERE SELF.U IS COMPUTED FROM PTS ADDEDF SO FAR), AND SELECTT CANDIDATE BASED ON UCB
        #print("Applying UCB Based Selection")
        ucbs = []

        if args.alternative:

            if args.thompson_sampling:
                print('Selecting Using Mahalanobis Based Thompson Sampling')
            else:
                print('Selecting Using Mahalanobis Based Posterior Computation')

            if learner.pool:
                if learner.ucb_wo_replacement:
                    unseen_indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                    #print('learner_unique_data:', learner.unique_data['ucb_replacement_graph_indices'])
                    if learner.pool_top_means:
                        unseen_indices_means = learner.data['means'][unseen_indices]
                        indices = unseen_indices[np.argpartition(unseen_indices_means, -learner.pool_num)[-learner.pool_num:]]
                    else:
                        indices = learner._rds.choice(unseen_indices, learner.pool_num, replace=False)

                    #print('Unseen pts:', unseen_indices.shape)
                    #print('Collected pts:', len(learner.unique_data['ucb_replacement_graph_indices']))
                else:
                    if learner.pool_top_means:
                        indices = unseen_indices[np.argpartition(learner.data['means'], -learner.pool_num)[-learner.pool_num:]]
                    else:
                        indices = learner._rds.choice(range(learner.num_actions), learner.pool_num, replace=False)  
            else:
                if learner.ucb_wo_replacement:
                    indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                else:
                    indices = np.arange(learner.num_actions)

            #print('Size of pool after eliminating repetitions:', len(indices))

            #kernel_matrix = torch.matmul(learner.G, learner.G.t()).to(dtype=learner.dtype).to(learner.device)
            #learner.U = torch.inverse(torch.diag(torch.ones(learner.G.shape[0]).to(learner.device) * learner.alg_lambda) \
            #                           + kernel_matrix).to(dtype=learner.dtype)

            if learner.initgrads_on_fly:
                g_vectors = torch.stack(learner.init_grads_on_demand(indices)).to(learner.device_select).to(dtype=learner.dtype)
            else:
                #print('g_vectors_device:', torch.stack(learner.init_grad_list_cpu)[indices].device)
                #g_vectors = torch.stack(learner.init_grad_list_cpu)[indices].to(learner.device_select).to(dtype=learner.dtype)
                g_vectors = learner.init_grad_list_cpu[indices,:].to(learner.device_select).to(dtype=learner.dtype)

                #print('g_vectors_device:', g_vectors.device)
                #print('g_vectors:', g_vectors)
                #print('g_vectors:', g_vectors.to(learner.device))

            if learner.initgrads_on_fly:
                g_vectors_observed = torch.stack(learner.init_grads_on_demand(learner.data['graph_indices'])).to(learner.device_select).to(dtype=learner.dtype)
            else:
                #print('g_vectors_observed_device:', torch.stack(learner.init_grad_list)[learner.data['graph_indices']].device)
                #g_vectors_observed = torch.stack(learner.init_grad_list_cpu)[learner.data['graph_indices']].to(learner.device_select).to(dtype=learner.dtype)
                g_vectors_observed = learner.init_grad_list_cpu[learner.data['graph_indices'],:].to(learner.device_select).to(dtype=learner.dtype)

                #print('g_vectors_observed_device:', g_vectors_observed.device)
                #3print('g_vectors_observed:', g_vectors_observed[0])

            #print('learner_unique_data:', learner.unique_data['ucb_replacement_graph_indices'])
            if len(learner.data['graph_indices']) <= 1:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=learner.dtype) / learner.neuron_per_layer)
                #kx_t = (torch.matmul(g, torch.stack(learner.init_grad_list)[:,learner.data['graph_indices']].flatten(start_dim = 1)) / learner.neuron_per_layer)
                post_vars = (learner.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g_vectors * g_vectors_observed, dim=1 ) / learner.neuron_per_layer - torch.sum(kx_t_matrix * learner.U.to(learner.device_select) * kx_t_matrix, dim=1))
            else:
                #print('Computing variances based on Mahalanobis Expression')
                kx_t_matrix = (torch.matmul(g_vectors, g_vectors_observed.t()).to(dtype=learner.dtype) / learner.neuron_per_layer)
                #print('kxt_device:', kx_t_matrix.device)
                #print('kxt:', kx_t_matrix)
                #print('kxt:', kx_t_matrix.shape)
                #kx_t = (torch.matmul(g, torch.stack(learner.init_grad_list)[:,learner.data['graph_indices']].flatten(start_dim = 1)) / learner.neuron_per_layer)
                #print(kx_t.shape)
                #print('self_U_device_before:', learner.U.device)
                #print('self_U:', learner.U)
                post_vars = (learner.alg_lambda)**(-0.5)*torch.sqrt(torch.sum( g_vectors * g_vectors, dim=1 ) / learner.neuron_per_layer - torch.sum(torch.matmul(kx_t_matrix, learner.U.to(learner.device_select)) * kx_t_matrix, dim=1))
                #print('post_vars_device:', post_vars.device)
                #print('post_vars:', post_vars)
                #print('self_U_device_after:', learner.U.device)
                #print('post_var:', post_vars.shape)
            #post_means = torch.tensor([learner.get_post_mean(indices[i]) for i in range(len(indices))])
            post_means = learner.data['means'][indices]
            #print('post_means_device:', post_means.device)
            #print('post_means:', post_means[0])
            if learner.bernoulli_selection:
                ber_param = 1/(len(learner.data['graph_indices'])-learner.num_pretrain_steps+1)
                coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                if coin_toss_result:
                    ucbs = post_means.to(learner.device_select)
                else:
                    if args.thompson_sampling:
                        print('Thompson')
                        ucbs = torch.tensor([np.random.normal(loc=m, scale=s, size=1) for m,s in zip(post_means, np.sqrt(args.exploration_coef)*post_vars)])
                    else:
                        ucbs = post_means.to(learner.device_select) + np.sqrt(learner.exploration_coef) * post_vars
            else:
                if args.thompson_sampling:
                    print('Thompson')
                    ucbs = torch.tensor([np.random.normal(loc=m, scale=s, size=1) for m,s in zip(post_means, np.sqrt(args.exploration_coef)*post_vars)])
                else:
                    ucbs = post_means.to(learner.device_select) + np.sqrt(learner.exploration_coef) * post_vars

            if learner.select_K_together:
                ix_pool = torch.topk(ucbs, learner.select_K).indices
            else:
                ix_pool = torch.argmax(ucbs).item()
                #print('ix_pool_device:', ix_pool)

            if learner.pool:
                ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]
            else:
                ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]
            #print('ix1:',ix)
            #print('ucbs1:', ucbs1)

        elif args.no_var_computation:

            print('Selecting Using No var computation')

            if not args.oracle:

                if learner.pool:

                    if learner.ucb_wo_replacement:
                        unseen_indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                        if learner.pool_top_means:
                            unseen_indices_means = learner.data['means'][unseen_indices]
                            indices = unseen_indices[np.argpartition(unseen_indices_means, -learner.pool_num)[-learner.pool_num:]]
                        else:
                            indices = learner._rds.choice(unseen_indices, learner.pool_num, replace=False)
                        #print('Unseen pts:', unseen_indices.shape)
                        #print('Collected pts:', len(learner.unique_data['ucb_replacement_graph_indices']))
                    else:
                        if learner.pool_top_means:
                            indices = unseen_indices[np.argpartition(learner.data['means'], -learner.pool_num)[-learner.pool_num:]]
                        else:
                            indices = learner._rds.choice(range(learner.num_actions), learner.pool_num, replace=False)  
                else:

                    if learner.ucb_wo_replacement:
                        indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                    else:
                        indices = np.arange(learner.num_actions)

                post_vars = torch.zeros(len(indices)).to(learner.device_select)
                post_means = learner.data['means'][indices]
                
                if args.rand:
                    print('Randomly Selecting')
                    if learner.select_K_together:
                        ix_pool = env_rds.choice(np.arange(len(indices)), size=learner.select_K)
                    else:
                        ix_pool = env_rds.choice(np.arange(len(indices)), size=1)[0]
                    if learner.pool:
                        ix = indices[ix_pool].tolist() if learner.select_K_together else indices[ix_pool]
                    else:
                        ix = indices[ix_pool].tolist() if learner.select_K_together else indices[ix_pool]

                else:
                    if learner.bernoulli_selection:
                        ber_param = 1/(len(learner.data['graph_indices'])-learner.num_pretrain_steps+1)
                        coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                        if coin_toss_result:
                            ucbs = post_means.to(learner.device_select)
                        else:
                            ucbs = post_means.to(learner.device_select) + np.sqrt(learner.exploration_coef) * post_vars
                    else:
                        ucbs = post_means.to(learner.device_select)

                    if learner.select_K_together:
                        ix_pool = torch.topk(ucbs, learner.select_K).indices
                    else:
                        ix_pool = torch.argmax(ucbs).item()
                    if learner.pool:
                        ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]
                    else:
                        ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]

            elif args.oracle:

                print('Selecting Using the Oracle')
                if learner.pool:
                    if learner.ucb_wo_replacement:
                        unseen_indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                        if learner.pool_top_means:
                            unseen_indices_means = learner.data['means'][unseen_indices]
                            indices = unseen_indices[np.argpartition(unseen_indices_means, -learner.pool_num)[-learner.pool_num:]]
                        else:
                            indices = learner._rds.choice(unseen_indices, learner.pool_num, replace=False)
                        #print('Unseen pts:', unseen_indices.shape)
                        #print('Collected pts:', len(learner.unique_data['ucb_replacement_graph_indices']))
                    else:
                        if learner.pool_top_means:
                            indices = unseen_indices[np.argpartition(learner.data['means'], -learner.pool_num)[-learner.pool_num:]]
                        else:
                            indices = learner._rds.choice(range(learner.num_actions), learner.pool_num, replace=False)  
                else:
                    if learner.ucb_wo_replacement:
                        indices = np.array(list(set(range(learner.num_actions)) - set(learner.unique_data['ucb_replacement_graph_indices'])))
                    else:
                        indices = np.arange(learner.num_actions)
                post_vars = torch.zeros(len(indices)).to(learner.device_select)
                #post_means = torch.tensor([learner.get_post_mean(i) for i in range(len(indices))])
                #post_means = torch.tensor(np.array(learner.QM9_Dataset.data.y)[indices,learner.reward]).to(learner.device)
                post_means = torch.tensor([learner.QM9_Dataset[i].y.reshape(-1,1)[0] for i in indices]).to(learner.device_select)
                #print('Post vars shape:', post_vars.shape)
                #print('Post means shape:', post_means.shape)
                if learner.bernoulli_selection:
                    ber_param = 1/(len(learner.data['graph_indices'])-learner.num_pretrain_steps+1)
                    coin_toss_result = np.random.choice([True,False],p=[ber_param,1-ber_param])
                    if coin_toss_result:
                        ucbs = post_means.to(learner._select)
                    else:
                        ucbs = post_means.to(learner.device_select) + np.sqrt(learner.exploration_coef) * post_vars
                else:
                    ucbs = post_means.to(learner.device_select) + np.sqrt(learner.exploration_coef) * post_vars

                if learner.select_K_together:
                    ix_pool = torch.topk(ucbs, learner.select_K).indices
                else:
                    ix_pool = torch.argmax(ucbs).item()

                if learner.pool:
                    ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]
                else:
                    ix = indices[ix_pool.cpu()].tolist() if learner.select_K_together else indices[ix_pool]

        return ix
    
    t0 = time.time()

    # run bandit algorithm
    regrets = []
    regrets_bp = []
    cumulative_regret = 0
    last_cumulative_regret = 0
    last_regret_bp = 0
    cumulative_regret_bp = 0
    rewards = []
    rewards_bp = []
    cumulative_reward = 0
    cumulative_reward_bp = 0
    new_indices = []
    new_rewards = []
    actions_all = []
    avg_vars = []
    pick_vars_all = []
    pick_rewards_all = []

    all_indices_seen = []
    seen_losses = []
    unseen_losses = []

    pooled_indices = []
    original_stop_count = learner.stop_count 

    """
    THE TRAINING LOOP
    """
    #print("Model's state_dict:")
    #for param_tensor in learner.func.state_dict():
        #print(param_tensor, "\t", learner.func.state_dict()[param_tensor].size())

    t0 = time.time()

    MAX_NUM_NODES = 29
    NUM_ACTIONS = args.num_actions

    index = list(np.arange(len(graph_data)))
    #print('Index:', index)

    def feat_pad(feat_mat):
        return torch.nn.functional.pad(feat_mat,pad=(0,0,0,MAX_NUM_NODES-len(feat_mat)), value=0)#value=float('nan'))
    
    def z_pad(feat_mat):
        return torch.nn.functional.pad(feat_mat,pad=(0,MAX_NUM_NODES-len(feat_mat)), value=0)# value=float('nan'))   
    
    def rand_jitter(arr):
        stdev = .001 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    features_list = []
    rewards_list = []

    for ix in index:

        #features_list.append(feat_pad(torch.tensor(graph_data[ix].x)).flatten())
        features_list.append(torch.cat((feat_pad(dataset[ix].x.float()), feat_pad(dataset[ix].pos.float()), z_pad(dataset[ix].z.float())[:,None]), 1).flatten())
        rewards_list.append(graph_rewards[ix]) 

    rewards_arr = np.array(rewards_list)
    rewards_arr.resize((len(graph_data),1))

    #for e in features_list:
        #print(e.shape)
    
    features_list = torch.stack(features_list).numpy().astype(np.float32)

    #print("Features_list_shape:", features_list.shape)
    #print('Type Features List:', type(features_list))

    #reducer = umap.UMAP(n_neighbors=int(NUM_ACTIONS/10), min_dist=0.2)
    reducer = umap.UMAP(n_neighbors=80, min_dist=0.2)
    embedding = reducer.fit_transform(features_list)

    t1 = time.time()
    print('Time for UMAP:', t1-t0)
    #embedding = []

    gif_count = 0
    colors_gif = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]

    if args.large_scale:
        if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
            pass
        else:
            os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
    else:
        if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
            pass
        else:
            os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')

    t = 0
    print_count = 1

    acq_time_per_step = []
    while t < args.T - 1: #T is the total number of steps that the algortihm runs for, see paper

        # if idx_search in learner.data['graph_indices']:
        #     print(f'15.7709 VALUE FOUND AT STEP {t}')


        '''
        TRAIN FROM SCRATCH (WITH POSSIBLY RANDOMLY SUBSAMPLING THE BATCHES EACH TIME) UNTIL T1 STEPS
        '''
        if args.T1 < args.pretrain_steps:
            warnings.warn("Train From scratch steps end before pretraining_steps, so no effect will be observed unless you use pretraining")

        if t <= args.T1:
            learner.train_from_scratch = True
        else:
            learner.train_from_scratch = False
        
        """
       T0 DETERMINES WHETHER BATCH_SIZE IS ACTIVE OR NOT, THUS WHETHER WE NEED RTO PARALLELIZE!!
        """

        if t < args.T0: 

            start = time.monotonic()

            """
            SELECTION PHASE
            """

            if  t > args.pretrain_steps:

                if len(set(all_indices_seen[-args.explore_threshold:])) == 1:
                    action_t = learner.explore_throw_in()
                else:
                    if t > args.T2:
                        action_t = learner.select() #Otherwise just run selection algortihm via GNN UCB
                    else:
                        action_t = learner.explore()
            else: #otherwise, explore!
                action_t = learner.explore() #Pure exploration in ~40 pretraining steps to mimic


            """
            COMPUTATIONS FOR PLOTS ETC.
            """

            if args.select_K_together:
                all_indices_seen.extend(action_t)
                actions_all.extend(action_t) #Iteratively create a list of proposed actions
            else:
                all_indices_seen.append(action_t)
                actions_all.append(action_t)

            observed_reward_t = evaluate(idx_list = [action_t] if not isinstance(action_t, list) else action_t , noisy=args.noisy_reward, reward_list=graph_rewards, noise_var=args.noise_var, _rds = env_rds)
            pick_rewards_all.append(observed_reward_t)

            """
            ADD DATA (POSTERIOR UPDATES)
            """

            learner.add_data([action_t] if not isinstance(action_t, list) else action_t, [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)

            if args.select_K_together:
                regret_t = [(max_reward - graph_rewards[a]).item() for a in action_t] #average (over noise) regret
                cumulative_regret += sum(regret_t)
                best_action_t = learner.exploit()
                regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
                cumulative_regret_bp += regret_t_bp
            else:
                regret_t = (max_reward - graph_rewards[action_t]).item()
                cumulative_regret += regret_t
                best_action_t = learner.exploit()
                regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
                cumulative_regret_bp += regret_t_bp

            regrets.append(cumulative_regret)
            regrets_bp.append(cumulative_regret_bp)
            if args.select_K_together:
                pick_vars_all.extend(learner.get_post_var_print_every(action_t))
            else:
                pick_vars_all.append(learner.get_post_var(action_t))

            """
            TRAIN GNN
            """

            # only train the network if you are passed pretrain time!!!
                
            if t > args.pretrain_steps:
                if args.batch_GD:
                    loss = learner.train_batch()
                else:
                    loss = learner.train()

            end = time.monotonic()

            elapsed = end - start
            acq_time_per_step.append(elapsed/args.select_k*1e3 if args.select_K_together else elapsed*1e3)
                    
        else:  # After some time just train in batches, as explained in paper appandix

            start = time.monotonic()

            """
            SELECTION PHASE
            """

            pool = mp.pool.ThreadPool(mp.cpu_count())#//2)
            print('CPU COUNT:',mp.cpu_count())

            actions_prll = []
            action_t = []

            if  t > args.pretrain_steps:
                if t > args.T2:

                    learner.compute_means_for_sel()

                    # with mp.Pool(processes=mp.cpu_count()) as pool:
                    #     actions = pool.apply_async(learner.select, range(args.batch_size))
                    #     action_t.append(actions.get())

                    #action_t = learner.select() #Otherwise just run selection algortihm via GNN UCB

                    #for _ in range(min(args.batch_size, args.print_every - print_count)):
                #chunksize = 5
                #for _ in range(chunksize):
                    #pool = mp.pool.ThreadPool(mp.cpu_count()//2)
                    #print('CPU COUNT:',mp.cpu_count())
                    for _ in range(args.batch_size):
                        #np.random.seed(args.seed+t+h)
                        pool.apply_async(run_select, args = (0,), callback = actions_prll.append, error_callback=handle_error)# callback = actions_prll.append)
                        #actions_prll = [result.get() for result in results]

                #action_t = pool.map(learner.select, list(range(args.batch_size)))
                    pool.close()
                    pool.join()

                   # action_t = learner.run_sel()
                    
                else:
                    # with mp.Pool(processes=mp.cpu_count()) as pool:
                    #     actions = pool.apply_async(learner.select, range(args.batch_size))
                    #     action_t.append(actions.get())

                    #action_t = learner.explore()
                    #pool = mp.pool.ThreadPool(mp.cpu_count()//2)
                    for _ in range(min(args.batch_size, args.print_every - print_count)):
                        pool.apply_async(run_explore, args = (0,), callback = actions_prll.append)

                    #action_t = pool.map(learner.explore, list(range(args.batch_size)))
                    pool.close()
                    pool.join()

                    #action_t = learner.run_exp()

            else: #otherwise, explore!
                #print('INSIED EXPLORE')
                # with mp.Pool(processes=mp.cpu_count()) as pool:
                #     actions = pool.apply_async(learner.select, range(args.batch_size))
                #     print('Pool Actions:', action_t)
                #     action_t.append(actions.get())
                
                #action_t = learner.explore() #Pure exploration in ~40 pretraining steps to mimic

                #pool = mp.pool.ThreadPool(mp.cpu_count()//2)
                for _ in range(min(args.batch_size, args.print_every - print_count)):
                    pool.apply_async(run_explore, args = (0,), callback = actions_prll.append, error_callback = handle_error)

                #action_t = pool.map(learner.explore, list(range(args.batch_size)))
                pool.close()
                pool.join()

                #action_t = learner.run_exp()



            print('ACTIONS PRLL:', actions_prll)
            action_t = actions_prll
            #print('ACTIONS PRLL:', actions_prll)
            
            #pool.join()

            """
            COMPUTATIONS FOR PLOTS ETC.
            """

            if args.select_K_together:
                all_indices_seen.extend(list(itertools.chain.from_iterable(action_t)))
                actions_all.extend(list(itertools.chain.from_iterable(action_t))) #Iteratively create a list of proposed actions
            else:
                all_indices_seen.extend(action_t)
                actions_all.extend(action_t)

            observed_reward_t = evaluate(idx_list = action_t if not isinstance(action_t[0], list) else list(itertools.chain.from_iterable(action_t)), noisy=args.noisy_reward, reward_list=graph_rewards, noise_var=args.noise_var, _rds = env_rds)
            pick_rewards_all.extend(observed_reward_t)

            """
            ADD DATA (POSTERIOR UPDATES)
            """

            if args.ucb_wo_replacement:  #Only to keep track of pts sampled at each step, no posterior updates!!
                learner.add_data_ucb_replacement(action_t if not isinstance(action_t[0], list) else list(itertools.chain.from_iterable(action_t)), [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)
                
            #learner.add_data(action_t if not isinstance(action_t[0], list) else list(itertools.chain.from_iterable(action_t)), [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)
            learner.add_data_prll(action_t if not isinstance(action_t[0], list) else list(itertools.chain.from_iterable(action_t)), [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)

            if args.select_K_together:
                regret_t = [(max_reward - graph_rewards[a]).item() for a in list(itertools.chain.from_iterable(action_t))] #average (over noise) regret
                for r in regret_t:
                    cumulative_regret += r

                    best_action_t = learner.exploit()
                    regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
                    cumulative_regret_bp += regret_t_bp

                    regrets.append(cumulative_regret)
                    regrets_bp.append(cumulative_regret_bp)

            else:
                last_cumulative_regret = regrets[-1] if t > 0 else 0
                regret_t = [(max_reward - graph_rewards[a]).item() for a in action_t] #average (over noise) regret
                #cumulative_regret += sum(regret_t)
                regrets.extend((np.cumsum(regret_t) + last_cumulative_regret).tolist())
                cumulative_regret = cumulative_regret + np.sum(regret_t) + last_cumulative_regret

                last_regret_bp = regrets_bp[-1] if t > 0  else 0
                regret_t_bp = np.empty(args.batch_size)
                best_action_t = learner.exploit()
                regret_t_bp.fill((max_reward - graph_rewards[best_action_t]).item())
                regrets_bp.extend((np.cumsum(regret_t_bp)+ last_regret_bp).tolist())
                cumulative_regret_bp = cumulative_regret_bp + np.sum(regret_t_bp) + last_regret_bp


                # for r in regret_t:
                #     cumulative_regret += r

                #     best_action_t = learner.exploit()
                #     regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
                #     cumulative_regret_bp += regret_t_bp

                #     regrets.append(cumulative_regret)
                #     regrets_bp.append(cumulative_regret_bp)

            if args.select_K_together:
                pick_vars_all.extend(learner.get_post_var_print_every(list(itertools.chain.from_iterable(action_t))))
            else:
                pick_vars_all.extend(learner.get_post_var_print_every(action_t))

            """
            TRAIN GNN
            """

            # only train the network if you are passed pretrain time
            if t > args.pretrain_steps:
                if args.batch_GD:
                    loss = learner.train_batch()
                else:
                    loss = learner.train()

            new_indices = []  
            new_rewards = []
            if args.pool_top_means:
                learner.data['means'] = np.array(learner.get_post_mean_print_every(range(args.num_actions))) #Update predicted means for pool_top_means based pooling

            end = time.monotonic()

            elapsed = end - start
            acq_time_per_step.append(elapsed/args.select_k/args.batch_size*1e3 if args.select_K_together else elapsed/args.batch_size*1e3)

        if t < args.T0:
            t += 1
            #print_count_final += 1


        if t >= args.T0:
            t += min(args.batch_size, args.print_every - print_count)
            #print_count_final += min(args.batch_size, args.print_every - print_count)
            #print('print_count_final:', print_count_final)
            print('t:',t)

        if t % args.print_every == 0:
            print('At step {}: Action{}, Regret {}'.format(t + 1, action_t, cumulative_regret))
            if args.runner_verbose:
                '''############################################# REGRETS&VARIANCES ################################################'''
                #pick_vars_all.extend((learner.get_post_var_print_every(learner.data['graph_indices'][-args.print_every:])).tolist())
                print('Verbose is true')
                print('At step {}: Action{}, Regret {}'.format(t + 1, action_t, cumulative_regret))
                # plot conf ests
                #means = np.array([learner.get_post_mean(idx) for idx in range(args.num_actions)])
                #vars = np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])

                vars_list = []
                range_list = range(args.num_actions)
                chunk_size = 200
                range_chunks = [range_list[i:i + chunk_size] for i in range(0, len(range_list), chunk_size)]
                for idxs in range_chunks:
                    temp = learner.get_post_var_print_every(idxs)
                    vars_list.extend(temp if isinstance(temp, list) else temp.cpu().tolist())
                #vars = np.array([vars_list.append(learner.get_post_var_print_every(idxs)) for idxs in range_chunks])
                vars = np.array(vars_list)
                print('Len vars:',len(vars))
                print('VARS:', vars)
                                                 
                means = np.array(learner.get_post_mean_print_every(range(args.num_actions)))
                #print('pick vars all', pick_vars_all)
                print('pick vars all len', len(pick_vars_all))
                avg_vars.append(np.mean(vars))
                if print_count > 0:
                    plt.figure(1)
                    plt_regret(regrets = regrets, regrets_bp = regrets_bp,net = args.net, t=t, print_every=args.print_every,plot_vars=True,avg_vars=avg_vars, pick_vars_all=pick_vars_all)
                    if args.large_scale:
                        # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                        #     pass
                        # else:
                        #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale/regrets.jpg')
                    else:
                        # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                        #     pass
                        # else:
                        #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                        plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale/regrets.jpg')
                    plt.close()

                    '''############################################# REWARDS(INVERSE REGRET) ################################################'''
                    # plt.figure(2)
                    # plt_cumulative_rew(rewards = rewards, rewards_bp = rewards_bp,net = args.net, t=t, print_every=args.print_every,plot_vars=True,avg_vars=avg_vars, pick_vars_all=pick_vars_all)
                    # if args.large_scale:
                    #     # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                    #     #     pass
                    #     # else:
                    #     #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                    #     plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale/rewards.jpg')
                    # else:
                    #     # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                    #     #     pass
                    #     # else:
                    #     #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                    #     plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale/rewards.jpg')
                    # plt.close()

                    #fig = plt.figure(figsize=(6, 6))
                    #axs=fig.add_axes([0,0,1,1])
                    #fig, axs = plt.subplots(4, 1)
                    '''############################################# EXPLORED POINTS SINCE LAST TIME ################################################'''
                    plt.figure(3)
                    if gif_count == 0:

                        plt.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        marker='o',
                        facecolors='none', 
                        edgecolors='black',
                        linewidth = 0.01,
                        s = 1/2)
                        #axs[0].title.set_text('UMAP projection of Action Set')

                    collected_embeddings = embedding[learner.data["graph_indices"][-args.print_every:],:]

                    plt.scatter(
                        collected_embeddings[:, 0],
                        collected_embeddings[:, 1],
                        c=colors_gif[int(gif_count%20)],
                        s = 1/2)
                    #plt.set_xlim([np.min(embedding[:, 0]-1), np.max(embedding[:,0])+1])
                    #plt.set_ylim([np.min(embedding[:, 1]-1), np.max(embedding[:, 1])+1])
                    plt.xlim([np.min(embedding[:, 0]-1), np.max(embedding[:,0])+1])
                    plt.ylim([np.min(embedding[:, 1]-1), np.max(embedding[:,1])+1])

                    #fig.savefig(f'../../explored_pts_{gif_count}', format='svg')
                    if args.large_scale:
                        # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                        #     pass
                        # else:
                        #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                        plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts_{gif_count}.svg', format='svg')
                        plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts_{gif_count}.png')
                        plt.show()
                        #plt.close()
                        command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts_{gif_count}.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts_'+str(gif_count)+'.pdf']
                        #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                        #output, error = process.communicate()
                        subprocess.run(command, stderr=subprocess.PIPE, text=True)
                    else:
                        # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                        #     pass
                        # else:
                        #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                        plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts_{gif_count}.svg', format='svg')
                        plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts_{gif_count}.png')
                        #plt.savefig(f'../../explored_pts_{gif_count}.svg', format='svg')
                        #plt.savefig(f'../../explored_pts_{gif_count}.png')
                        plt.show()
                        #plt.close()
                        command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts_{gif_count}.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts_'+str(gif_count)+'.pdf']
                        #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                        #output, error = process.communicate()
                        subprocess.run(command, stderr=subprocess.PIPE, text=True)

                    gif_count+=1

        if t >= args.T-1: #This way saves plot only in final form  
            vars_list = []
            range_list = range(args.num_actions)
            chunk_size = 200
            range_chunks = [range_list[i:i + chunk_size] for i in range(0, len(range_list), chunk_size)]
            for idxs in range_chunks:
                temp = learner.get_post_var_print_every(idxs)
                vars_list.extend(temp if isinstance(temp, list) else temp.cpu().tolist())
            #vars = np.array([vars_list.append(learner.get_post_var_print_every(idxs)) for idxs in range_chunks])
            vars = np.array(vars_list)     

            means = np.array(learner.get_post_mean_print_every(range(args.num_actions)))

            print(learner.unique_data['graph_indices'])
            print(learner.data['graph_indices'])
            print("Max Reward Index:", np.argmax(graph_rewards))
            print('Vars:', vars)
            plot_indices = np.arange(args.num_actions)
            '''############################################# MEANS&VARIANCES ################################################'''
            plt.figure()
            print('plot')
            random_indices = np.random.choice(means.shape[0], size=200, replace=False)
            plt.plot(means[random_indices], '-', label='means', color='#9dc0bc', linewidth=0.6)
            #plt.scatter(plot_indices, means, '-', label='means', color='#9dc0bc')
            #plt.title(r'$Confidence\ and\ Mean\ Estimates$')
            plt.fill_between(np.arange(random_indices.shape[0]), means[random_indices] - np.sqrt(args.exploration_coef) * vars[random_indices],
                                        means[random_indices] + np.sqrt(args.exploration_coef) * vars[random_indices], alpha=0.3, color='#ffa500')
            plt.plot(np.array(graph_rewards)[random_indices], label='true function', color='#7c7287', linewidth=0.6)
            #plt.scatter(plot_indices, graph_rewards, label='true function', color='#7c7287')
            color = [item * 255 / (t + 1) for item in np.arange(t + 1)]
            included_indices = np.isin(random_indices, np.array(actions_all))
            included_indices_actions_all = []
            for i in range(len(included_indices)):
                if included_indices[i] == True:
                    included_indices_actions_all.append(i)
            #color_small = [item * 255 / (len(included_indices_actions_all) + 1) for item in range(len(included_indices.shape))]
            plt.scatter(included_indices_actions_all,
                                evaluate(idx_list=included_indices_actions_all, noisy=False, reward_list=np.array(graph_rewards)[random_indices], noise_var=args.noise_var,
                                            _rds=env_rds), s=1/4)
            
            print(len(color))
            plt.set_cmap('magma')
            plt.legend()
            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/meanandvariances.svg', format='svg')
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/meanandvariances.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/meanandvariances.pdf']
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/meanandvariances.svg', format='svg')
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/meanandvariances.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/meanandvariances.pdf']
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            plt.close()
            #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            #output, error = process.communicate()
            '''############################################# UMAP-REWARDS,SELECTED PTS, CONFIDENCES ################################################'''
            #print('Graph Data:', graph_data)
            collected_indices = learner.data["graph_indices"]
            #print('Collected indices:',collected_indices)
            fig, axs = plt.subplots(4, 1)

            axs[0].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c='black',
                s = 3)
            axs[0].title.set_text('UMAP projection of Action Set')

            collected_embeddings = embedding[collected_indices]

            pcm1 = axs[0].scatter(
                rand_jitter(collected_embeddings[:, 0]),
                rand_jitter(collected_embeddings[:, 1]),
                c=np.arange(0, collected_embeddings.shape[0]),
                cmap='gist_ncar',
                marker = "x",
                s = 2,
                alpha=0.5)
            axs[0].set_xlim([np.min(collected_embeddings[:, 0]-1), np.max(collected_embeddings[:,0])+1])
            axs[0].set_ylim([np.min(collected_embeddings[:, 1]-1), np.max(collected_embeddings[:, 1])+1])
            fig.colorbar(pcm1, ax=axs[0])

            pcm2 = axs[1].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=graph_rewards,
                cmap='plasma',
                marker = "s",
                s = 2)
            axs[1].title.set_text('Action Set Rewards',)
            axs[1].set_xlim([np.min(collected_embeddings[:, 0])-1, np.max(collected_embeddings[:,0])+1])
            axs[1].set_ylim([np.min(collected_embeddings[:, 1])-1, np.max(collected_embeddings[:, 1])+1])
            fig.colorbar(pcm2, ax=axs[1])

            #conf_bounds = np.sqrt(args.exploration_coef)*np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
            conf_bounds = np.sqrt(args.exploration_coef)*np.array(vars)
            pcm3 = axs[2].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=conf_bounds,
                cmap='plasma',
                marker = "s",
                s = 2)
            axs[2].title.set_text('Action Set Final Confidences',)
            axs[2].set_xlim([np.min(collected_embeddings[:, 0])-1, np.max(collected_embeddings[:,0])+1])
            axs[2].set_ylim([np.min(collected_embeddings[:, 1])-1, np.max(collected_embeddings[:, 1])+1])
            fig.colorbar(pcm3, ax=axs[2])

            visit_frequencies = np.histogram(learner.data['graph_indices'], bins = np.arange(1, np.max(args.num_actions)+2))[0]
            pcm4 = axs[3].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=visit_frequencies,
                cmap='gist_ncar',
                marker = "x",
                s = 2)
            axs[3].title.set_text('Action Set Explored/Selected Pts',)
            axs[3].set_xlim([np.min(embedding[:, 0]-1), np.max(embedding[:,0])+1])
            axs[3].set_ylim([np.min(embedding[:, 1]-1), np.max(embedding[:, 1])+1])
            fig.colorbar(pcm4, ax=axs[3])

            fig.set_size_inches(20.5, 20.5)
            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                fig.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/collected_pts.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(),f'./plots_bartu/{args.reward_plot_dir}/large_scale/collected_pts.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/collected_pts.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                fig.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/collected_pts.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(),f'./plots_bartu/{args.reward_plot_dir}/small_scale/collected_pts.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/collected_pts.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            '''############################################# CORRELATION OF CONF&MSE ################################################'''
            # plt.figure()
            # post_vars = args.exploration_coef*torch.tensor(np.array(vars))
            # mse_loss = torch.nn.MSELoss(reduction='none')
            # rews = torch.tensor(graph_rewards)
            # mean_squared_errors = np.array(mse_loss(torch.tensor(means), rews))

            # #max_val = torch.max(torch.tensor([torch.max(torch.tensor(post_vars).flatten()), torch.max(torch.tensor(mean_squared_errors).flatten())]))
            # #min_val = torch.min(torch.tensor([torch.min(torch.tensor(post_vars).flatten()), torch.min(torch.tensor(mean_squared_errors).flatten())]))
            # max_val_vars = torch.tensor(torch.max(torch.tensor(post_vars).flatten()))
            # max_val_mse = torch.tensor(torch.max(torch.tensor(mean_squared_errors).flatten()))
            # min_val_vars = torch.tensor(torch.min(torch.tensor(post_vars).flatten()))
            # min_val_mse = torch.tensor(torch.min(torch.tensor(mean_squared_errors).flatten()))

            # plt.scatter(mean_squared_errors.flatten(), post_vars.flatten(), s=1/2)
            # plt.plot([min_val_mse, max_val_mse], [min_val_vars, max_val_vars], alpha=0.3)
            # plt.grid(alpha=0.3)
            # plt.title(r'$Correlation\ of\ Confidence\ Intervals\ and\ MSEs:$' + '{:.3f}'.format(pearsonr(mean_squared_errors.flatten(), post_vars.flatten())[0]))
            # plt.colorbar()
            # plt.legend()
            # plt.xlabel(r'$MSEs$')
            # plt.ylabel(r"$Confidence\ Intervals$")

            # if args.large_scale:
            #     # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
            #     #     pass
            #     # else:
            #     #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
            #     plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/conf_interval_mse_correlation.svg', format='svg')
            #     plt.close()
            #     command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/conf_interval_mse_correlation.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/conf_interval_mse_correlation.pdf']
            #     #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            #     #output, error = process.communicate()
            #     subprocess.run(command, stderr=subprocess.PIPE, text=True)
            # else:
            #     # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
            #     #     pass
            #     # else:
            #     #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
            #     plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/conf_interval_mse_correlation.svg', format='svg')
            #     plt.close()
            #     command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/conf_interval_mse_correlation.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/conf_interval_mse_correlation.pdf']
            #     #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            #     #output, error = process.communicate()
            #     subprocess.run(command, stderr=subprocess.PIPE, text=True)


        
            '''############################################# MEANS_VS_REWARDS ################################################'''
            max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards).flatten()), torch.max(torch.tensor(means).flatten())]))
            min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards).flatten()), torch.min(torch.tensor(means).flatten())]))
            #print(max_val)
            plt.figure()
            #conf_bounds = np.sqrt(args.exploration_coef)*np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
            conf_bounds = np.sqrt(args.exploration_coef)*np.array(vars)
            cmap = matplotlib.colors.ListedColormap(['red', 'green'])
            names = [r'$Not\ Acquired$', r'$Acquired$']
            #plt.errorbar(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), xerr=conf_bounds, fmt='o', alpha=0.2)
            scatter = plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), \
                        c=np.isin(np.arange(args.num_actions), learner.data['graph_indices']), cmap=cmap, s=1/2)
            plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
            plt.grid(alpha=0.3)
            #plt.title(r'$Seen\ Samples\ vs\ Unseen\ Samples$')
            plt.legend(handles=scatter.legend_elements()[0], labels=names)
            #plt.colorbar()
            #plt.legend()
            plt.xlabel(r'$Predicted$')
            plt.ylabel(r"$True\ Reward$")

            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/means_vs_rewards_seen_vs_unseen.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/means_vs_rewards_seen_vs_unseen.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/means_vs_rewards_seen_vs_unseen.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/means_vs_rewards_seen_vs_unseen.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/means_vs_rewards_seen_vs_unseen.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/means_vs_rewards_seen_vs_unseen.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)

            '''############################################# TOP_K METRIC ################################################'''
            top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]

            top_k_num_samples = [int(p*args.num_actions/100.0) for p in top_k_percentages]

            sorted_action_space = np.argsort(-np.array(graph_rewards))
            print('Sorted_action_space:', sorted_action_space.shape)

            top_k_indices = []
            for i in top_k_num_samples:
                #top_k_indices.append(np.argpartition(train_dataset.data.y, -i)[-i:])
                top_k_indices.append(sorted_action_space[:i])

            collected_num_samples = []
            for indices in top_k_indices:
                collected_indices = set(indices).intersection(learner.data['graph_indices'])
                uncollected_indices = np.array(list(set(indices) - set(learner.data['graph_indices'])))
                print('collected:', len(collected_indices))
                print('uncollected:', len(uncollected_indices))
                print('total:', len(indices))
                collected_num_samples.append(len(collected_indices)/len(indices))

            plt.figure()
            plt.plot(top_k_percentages, collected_num_samples, linestyle='--', marker='o')
            plt.grid(alpha=0.3)
            #plt.title(r'$Percentages\ of\ Top-K\ Percentile\ Acquired$')
            plt.xlabel(r'$Top-K\ Percent\ Samples\ Queried$')
            plt.ylabel(r"$Percentage\ of\ Top-K\ Samples\ Acquired$")

            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/top_k_percentages_collected.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/top_k_percentages_collected.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/top_k_percentages_collected.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/top_k_percentages_collected.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/top_k_percentages_collected.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/top_k_percentages_collected.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)

            '''############################################# CALIBRATION AND SHARPNESS ################################################'''
            coverages = []
            alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for a in alpha:
                intervals = [st.norm.interval(a, loc=m, scale=s) for m,s in zip(means, args.exploration_coef*vars)]
                inside = [bound[0] <= r <= bound[1] for r,bound in zip(graph_rewards,intervals)]
                coverages.append(float(sum(inside))/float(len(inside)))
            
            alpha_sharpness = [0.5, 0.95]
            alpha_sharpness_str = [str(x) for x in alpha_sharpness]
            avg_widths = []
            for a in alpha_sharpness:
                intervals = [st.norm.interval(a, loc=m, scale=s) for m,s in zip(means, args.exploration_coef*vars)]
                widhts = [abs(bound[0] - bound[1]) for bound in intervals]
                #print('widths:', widths')
                avg_widths.append(np.mean(widhts))
                print('avg_widths:', avg_widths)

            plt.figure()
            plt.plot(alpha, coverages, linestyle='--', marker='o')
            plt.plot([0.0,0.0],[1.0,1.0],alpha=0.3)
            plt.grid(alpha=0.3)
            #plt.title(r'$Percentages\ of\ Top-K\ Percentile\ Acquired$')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r'$Calibration\ of\ Confidence\ Intervals$')

            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/calibration.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/calibration.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/calibration.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/calibration.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/calibration.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/calibration.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            
            plt.figure()
            plt.bar(alpha_sharpness_str, avg_widths, color='maroon', width=0.3)
            plt.grid(alpha=0.3)
            #plt.title(r'$Percentages\ of\ Top-K\ Percentile\ Acquired$')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r'$Sharpness\ of\ Confidence\ Intervals$')

            if args.large_scale:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/large_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/large_scale/sharpness.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/large_scale/sharpness.svg'), f'./plots_bartu/{args.reward_plot_dir}/large_scale/sharpness.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)
            else:
                # if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale'):
                #     pass
                # else:
                #     os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots_bartu/{args.reward_plot_dir}/small_scale')
                plt.savefig(f'./plots_bartu/{args.reward_plot_dir}/small_scale/sharpness.svg', format='svg')
                plt.close()
                command = ['svg42pdf', os.path.join(os.getcwd(), f'./plots_bartu/{args.reward_plot_dir}/small_scale/sharpness.svg'), f'./plots_bartu/{args.reward_plot_dir}/small_scale/sharpness.pdf']
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                #output, error = process.communicate()
                subprocess.run(command, stderr=subprocess.PIPE, text=True)

            '''############################################# EXPLORED PTS GIF ################################################'''
            frames = []
            for k in range(gif_count):
                if args.large_scale:
                    image = imageio.v2.imread(f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts_{k}.png')
                else:
                    image = imageio.v2.imread(f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts_{k}.png')
                frames.append(image)

            if args.large_scale:
                imageio.mimsave(f'./plots_bartu/{args.reward_plot_dir}/large_scale/explored_pts.gif', frames, duration=2) 
            else:
                imageio.mimsave(f'./plots_bartu/{args.reward_plot_dir}/small_scale/explored_pts.gif', frames, duration=2) 
        
        if t < args.T0:
            #t += 1
            #print_count_final += 1
            print_count += 1
            #print('t:', t)
            #print('print count:', print_count)

        if t >= args.T0:
            #t += min(args.batch_size, args.print_every - print_count)
            #print_count_final += min(args.batch_size, args.print_every - print_count)
            temp_print_count = print_count
            print_count += min(args.batch_size, args.print_every - temp_print_count)
            print('t:', t)
            print('print count:', print_count)
        
        if print_count % args.print_every == 0:
            print_count = 0

    collected_global_max = np.isin(max_reward_index, actions_all)
    graph_rewards = [a.item() for a in graph_rewards]
    pick_rewards_all = [a[0] if isinstance(a, list) else a.item() for a in pick_rewards_all]
    print('actions_all:', actions_all)
    print('graph_rews:', graph_rewards)
    print('means:', means)
    print('covereages:', coverages)
    print('avg vars:', avg_vars)
    print('time:', acq_time_per_step)
    print('rewards:', pick_rewards_all)
    print('regrets:', regrets)
    print('topk:', collected_num_samples)
    print('regrets_bp:', regrets_bp)
    print('colleceted_global_max:', collected_global_max)
    print('pick_vars_all:', pick_vars_all)
    if args.runner_verbose:
        print(f'{learner.name} with {args.T} steps takes {(time.time() - t0)/60} mins.')
    exp_results = {'actions': actions_all if isinstance(actions_all, list) else actions_all.tolist(),
    'rewards': pick_rewards_all if isinstance(pick_rewards_all, list) else pick_rewards_all.tolist(),
    'regrets': regrets if isinstance(regrets, list) else regrets.tolist(),
    'regrets_bp': regrets_bp if isinstance(regrets_bp, list) else regrets_bp.tolist(),
    'pick_vars_all': pick_vars_all if isinstance(pick_vars_all, list) else pick_vars_all.tolist(), 
    'avg_vars': avg_vars if isinstance(avg_vars, list) else avg_vars.tolist(), 
    'top_k':collected_num_samples, 
    'coverages':coverages, 
    'avg_widths':avg_widths, 
    'collected_max':collected_global_max,
    'graph_rewards':graph_rewards,
    'means':means,
    'indices':learner.data['graph_indices'],
    'time': acq_time_per_step}

    results_dict = {
        'exp_results': exp_results,
        'params': args.__dict__,
        'duration_total': (time.time() - t0)/60,
        'algorithm': 'ucb',
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        exp_hash = str(abs(json.dumps(results_dict['params'], sort_keys=True).__hash__()))
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)
        print('Duration:', (time.time() - t0)/60)

if __name__ == '__main__':

    #mp.set_sharing_strategy('file_system')
    mp.set_start_method('forkserver', force=True)
    #mp.set_start_method('spawn', force=True)

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

    parser.add_argument('--remove_smiles_for_sober', default=False, action='store_true', help='Whether to remove the unique smiles from QM9 (wrt github version) including or not including overlaps or high rews depending on which csv is read')

    parser.add_argument('--thompson_sampling', default=False, action='store_true', help='compute vars by alternative but select using thompson sampling rather than ucb')

    parser.add_argument('--rand', default=False, action='store_true', help='Used together with no_var_computation, runs GNN-SS-Rand')

    args = parser.parse_args()
    main(args)