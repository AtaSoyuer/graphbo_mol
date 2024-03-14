import os
import sys
import json
from algorithms_pyg import GnnUCB
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
from sklearn.model_selection import train_test_split
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import QM9
import matplotlib.colors as cm
import imageio
matplotlib.use('agg')
sys.path.append(os.path.abspath("./plot_scripts/")) 
import bundles
import scipy.stats as st
from scipy.stats.stats import pearsonr
from supervised_pretraining import SupervisedPretrain



def evaluate(idx_list: list, reward_list: list, noisy: bool, _rds , noise_var: float): #Create reward array from given list of indices and add aleotoric noise if need be
    #print('indices:',idx_list)
    rew = np.array([reward_list[idx] for idx in idx_list])
    #if noisy:
        #rew = rew + _rds.normal(0, noise_var, size=rew.shape)
    return list(rew)

def main(args):

    
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

        subset_indices = np.random.choice(len(dataset_without_pretraining), args.num_actions, replace=False)

        mask=np.full(len(dataset_without_pretraining),True,dtype=bool)
        mask[subset_indices]=False
        dataset_removed=dataset_without_pretraining[mask]
        print('Dataset removed:', len(dataset_removed))

        #val_subset_indices = np.random.choice(len(dataset_removed), int(args.num_actions))
        val_subset_indices = np.random.choice(len(dataset_removed), 100, replace=False)

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
                     select_K_together=args.select_K_together, select_K = args.select_K, random_state=algo_rds)

    #Initialize the algortihm & the model as above with learner 

    t0 = time.time()

    # run bandit algorithm
    regrets = []
    regrets_bp = []
    cumulative_regret = 0
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

    for t in range(args.T): #T is the total number of steps that the algortihm runs for, see paper
        '''
        TRAIN FROM SCRATCH (WITH POSSIBLY RANDOMLY SUBSAMPLING THE BATCHES EACH TIME) UNTIL T1 STEPS
        '''
        if args.T1 < args.pretrain_steps:
            warnings.warn("Train From scratch steps end before pretraining_steps, so no effect will be observed unless you use pretraining")

        if t <= args.T1:
            learner.train_from_scratch = True
            #learner.subsample = True #TODO:COMMENT????
            #learner.stop_count = learner.subsample_num #TODO:COMMENT????
        else:
            learner.train_from_scratch = False
            #learner.subsample = False #TODO:COMMENT????
            #learner.stop_count = original_stop_count #TODO:COMMENT????
        #print(learner.train_from_scratch)
        # only maximize ucb if you are passed pretrain time
        if  t > args.pretrain_steps:
            """
            THROWS IN RANDOMYL EXPLORED PTS IF STUCK ON A PT
            """
            if len(set(all_indices_seen[-args.explore_threshold:])) == 1:
                action_t = learner.explore_throw_in()
            #elif args.pool:
                #pooled_indices = env_rds_choice.choice(range(args.num_actions), args.pool_num, replace=False)
                #print(f"Pooling {args.pool_num} samples out of actionset of size {len(range(args.num_actions))}")
                #print('Pooled indices:', pooled_indices)
                #action_t = learner.select_pool(pool=pooled_indices)
                #action_t = learner.select()
                #action_t2 = learner.select_pool(pool=range(args.num_actions))
                #print('Actiont:', action_t)
                #print('Actiont2:', action_t2)
            else:
                if t > args.T2:
                    action_t = learner.select() #Otherwise just run selection algortihm via GNN UCB
                else:
                    action_t = learner.explore()
        else: #otherwise, explore!
            action_t = learner.explore() #Pure exploration in ~40 pretraining steps to mimic
            #some pretraining

        if args.select_K_together:
            all_indices_seen.extend(action_t)
            actions_all.extend(action_t) #Iteratively create a list of proposed actions
        else:
            all_indices_seen.append(action_t)
            actions_all.append(action_t)

        #The observed reward is actually a noisy version by problem formulation
        observed_reward_t = evaluate(idx_list = [action_t] if not isinstance(action_t, list) else action_t , noisy=args.noisy_reward, reward_list=graph_rewards, noise_var=args.noise_var, _rds = env_rds)
        pick_rewards_all.append(observed_reward_t)

        if args.select_K_together:
            regret_t = [(max_reward - graph_rewards[a]).item() for a in action_t] #average (over noise) regret
            cumulative_regret += sum(regret_t)
            best_action_t = learner.exploit()
            #print('best action:', best_action_t)
            #best_action_t = learner.best_predicted()
            regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
            cumulative_regret_bp += regret_t_bp
        else:
            regret_t = (max_reward - graph_rewards[action_t]).item()
            cumulative_regret += regret_t
            best_action_t = learner.exploit()
            regret_t_bp = (max_reward - graph_rewards[best_action_t]).item()
            cumulative_regret_bp += regret_t_bp
            
        #BP regret: I GUESS BEST POSSIBLE REGRET BY CHOOSING THE BEST SEEN REWARD SO DAR#OUT OF ALL THE 
        #PREVOOUSLY PROPOSED ACTIONS, PURE EXPLOITATION

        # reward_t = [graph_rewards[a].item() for a in action_t]
        # cumulative_reward += sum(reward_t)
        # reward_t_bp = graph_rewards[best_action_t].item()
        # cumulative_reward_bp += reward_t_bp

        if t < args.T0:
            learner.add_data([action_t] if not isinstance(action_t, list) else action_t, [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)
            # only train the network if you are passed pretrain time!!!
            if t > args.pretrain_steps:
                if args.batch_GD:
                    loss = learner.train_batch()
                else:
                    loss = learner.train()
            
            #NOTE:REMOVE IF IT DOESNT WORK!!!
            #else:
                #loss = learner.pretrain()
        else:  # After some time just train in batches, as explained in paper appandix
            # save the new datapoints
            if len(new_rewards) > 0:
                if args.select_K_together:
                    new_rewards.extend(observed_reward_t)
                    new_indices.extend(action_t)
                else:
                    new_rewards.append(observed_reward_t)
                    new_indices.append(action_t)
            else:
                new_rewards = [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t
                new_indices = [action_t] if not isinstance(action_t, list) else action_t
            # when there's enough, update the GP
            if args.ucb_wo_replacement:  #Only to keep track of pts sampled at each step, no posterior updates!!
                learner.add_data_ucb_replacement([action_t] if not isinstance(action_t, list) else action_t, [observed_reward_t] if not isinstance(observed_reward_t, list) else observed_reward_t)
            if t % args.batch_size == 0:
                learner.add_data(new_indices, new_rewards)
                # only train the network if you are passed pretrain time
                if t > args.pretrain_steps:
                    if args.batch_GD:
                        loss = learner.train_batch()
                    else:
                        loss = learner.train()
                #NOTE:REMOVE IF IT DOESNT WORK!!!
                #else:
                    #loss = learner.pretrain()
                new_indices = []  # remove from unused points
                new_rewards = []
                if args.pool_top_means:
                    learner.data['means'] = np.array(learner.get_post_mean_print_every(range(args.num_actions))) #Update predicted means for pool_top_means based pooling
                    #plot mean and variance estimates
        regrets.append(cumulative_regret)
        regrets_bp.append(cumulative_regret_bp)
        if args.select_K_together:
            pick_vars_all.extend(learner.get_post_var_print_every(action_t))
        else:
            pick_vars_all.append(learner.get_post_var(action_t))
        # rewards.append(cumulative_reward)
        # rewards_bp.append(cumulative_reward_bp)

        if t % args.print_every == 0:
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
                    vars_list.extend(learner.get_post_var_print_every(idxs).cpu())
                #vars = np.array([vars_list.append(learner.get_post_var_print_every(idxs)) for idxs in range_chunks])
                vars = np.array(vars_list)
                print('Len vars:',len(vars))
                                                 
                means = np.array(learner.get_post_mean_print_every(range(args.num_actions)))

                avg_vars.append(np.mean(vars))
                if t > 0:
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

        if t == args.T-1: #This way saves plot only in final form       

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
            
            # plt.figure()
            # plt.plot(seen_losses, '-', label='seen', color='#9dc0bc')
            # plt.title('Losses per Steps (After #Pretraining Steps)')
            # plt.plot(unseen_losses, label='unseen', color='#7c7287')
            # plt.savefig('seen_and_unseen_losses.svg', format = 'svg')
            # plt.close()
            # command = 'svg42pdf seen_and_unseen_losses.svg seen_and_unseen_losses.pdf'
            # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()
    collected_global_max = np.isin(max_reward_index, actions_all)
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
    'graph_rewards':[a.tolist() for a in graph_rewards],
    'means':means}

    results_dict = {
        'exp_results': exp_results,
        'params': args.__dict__,
        'duration_total': (time.time() - t0)/60,
        'algorithm': 'ucb'
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
    main(args)