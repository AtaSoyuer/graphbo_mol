import os
import json
from algorithms import GnnUCB
import time
import numpy as np
import torch
from utils_exp import NumpyArrayEncoder, read_dataset
from plot_scripts.utils_plot import plt_regret
from matplotlib import pyplot as plt
import matplotlib
import argparse
import umap 
import subprocess
import warnings



def evaluate(idx_list: list, reward_list: list, noisy: bool, _rds , noise_var: float): #Create reward array from given list of indices and add aleotoric noise if need be
    rew = np.array([reward_list[idx] for idx in idx_list])
    if noisy:
        rew = rew + _rds.normal(0, noise_var, size=rew.shape)
    return list(rew)

def main(args):

    print('Args:',args)
    
    #print('Run setting:', args.__dict__)

    # read full data
    env_rds = np.random.RandomState(args.seed)
    env_rds_choice = np.random.Generator(np.random.PCG64(args.seed)) #Gnumpy random Generator which is supposed to be faster
    graph_data_full, graph_rewards_full = read_dataset(args,env_rds) #From utils.py

    # Pick the data: The entire dataset has 10,000 graphs. Pick a random set of points to work with.
    indices = env_rds.choice(range(len(graph_data_full)), args.num_actions)
    '''
    THIS HERE LIMITS THE READ DATASET TO NUM_ACTIONS MANY GRAPHS INTOTAL
    '''

    graph_data = [graph_data_full[i] for i in indices]
    graph_rewards = [graph_rewards_full[i] for i in indices]

    max_reward = np.max(graph_rewards) #The maximum that we "dont know of"
    # set bandit algorithm
    assert args.num_actions == len(graph_data)
    assert len(graph_data) == len(graph_rewards)
    algo_rds = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    learner = GnnUCB(net = args.net, feat_dim = args.feat_dim, num_nodes = args.num_nodes,num_actions = args.num_actions, action_domain = graph_data, verbose=args.runner_verbose,
                     alg_lambda = args.alg_lambda, exploration_coef = args.exploration_coef, train_from_scratch=args.train_from_scratch, nn_aggr_feat=args.nn_aggr_feat,
                     num_mlp_layers = args.num_mlp_layers_alg, neuron_per_layer = args.neuron_per_layer, lr = args.lr, nn_init_lazy=args.nn_init_lazy, stop_count=args.stop_count, 
                     relative_improvement=args.relative_improvement, small_loss=args.small_loss, load_pretrained=args.load_pretrained, explore_threshold=args.explore_threshold, 
                     dropout=args.dropout, dropout_prob=args.dropout_prob, subsample=args.subsample, subsample_method=args.subsample_method, subsample_num=args.subsample_num, greedy=args.greedy, 
                     online=args.online_cov, complete_cov_mat=args.complete_cov_mat, alternative=args.alternative, GD_batch_size = args.GD_batch_size, batch_GD = args.batch_GD, random_state=algo_rds)

    #Initialize the algortihm & the model as above with learner 

    t0 = time.time()

    # run bandit algorithm
    regrets = []
    regrets_bp = []
    cumulative_regret = 0
    cumulative_regret_bp = 0
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
            elif args.pool:
                pooled_indices = env_rds_choice.choice(range(args.num_actions), args.pool_num, replace=False)
                print(f"Pooling {args.pool_num} samples out of actionset of size {len(range(args.num_actions))}")
                #print('Pooled indices:', pooled_indices)
                action_t = learner.select_pool(pool=pooled_indices)
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
        all_indices_seen.append(action_t)
        actions_all.append(action_t) #Iteratively create a list of proposed actions
        #The observed reward is actually a noisy version by problem formulation
        observed_reward_t = evaluate(idx_list = [action_t], noisy=args.noisy_reward, reward_list=graph_rewards, noise_var=args.noise_var, _rds = env_rds)
        pick_rewards_all.append(observed_reward_t)
        regret_t = max_reward - graph_rewards[action_t] #average (over noise) regret
        cumulative_regret += regret_t
        #BP regret: I GUESS BEST POSSIBLE REGRET BY CHOOSING THE BEST SEEN REWARD SO DAR#OUT OF ALL THE 
        #PREVOOUSLY PROPOSED ACTIONS, PURE EXPLOITATION
        best_action_t = learner.exploit()
        #best_action_t = learner.best_predicted()
        regret_t_bp = max_reward - graph_rewards[best_action_t]
        cumulative_regret_bp += regret_t_bp

        if t < args.T0:
            learner.add_data([action_t], [observed_reward_t])
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
                new_rewards.append(observed_reward_t)
                new_indices.append(action_t)
            else:
                new_rewards = [observed_reward_t]
                new_indices = [action_t]
            # when there's enough, update the GP
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
                #plot mean and variance estimates
        regrets.append(cumulative_regret)
        regrets_bp.append(cumulative_regret_bp)
        pick_vars_all.append(learner.get_post_var(action_t))

        """
        TO PLOT TRAINING LOSSES FOR SEENA ND UNSEEN PTS SO FAR
        """
        if t > args.pretrain_steps:

            all_data = np.arange(args.num_actions)
            seen_data = all_data[np.isin(np.arange(args.num_actions), learner.data['graph_indices'])]
            unseen_data = all_data[np.logical_not(np.isin(np.arange(args.num_actions), learner.data['graph_indices']))]

            seen_means = np.array([learner.get_post_mean(idx) for idx in seen_data])
            unseen_means = np.array([learner.get_post_mean(idx) for idx in unseen_data])

            seen_rewards = np.array([graph_rewards[idx] for idx in seen_data])
            unseen_rewards = np.array([graph_rewards[idx] for idx in unseen_data])

            seen_losses.append(torch.nn.functional.mse_loss(torch.tensor(seen_means), torch.tensor(seen_rewards)))
            unseen_losses.append(torch.nn.functional.mse_loss(torch.tensor(unseen_means), torch.tensor(unseen_rewards)))

        if t % args.print_every == 0:
            if args.runner_verbose:
                print('Verbose is true')
                print('At step {}: Action{}, Regret {}'.format(t + 1, action_t, cumulative_regret))
                # plot conf ests
                means = np.array([learner.get_post_mean(idx) for idx in range(args.num_actions)])
                vars = np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
                avg_vars.append(np.mean(vars))
                if t > 0:
                    plt.figure()
                    plt_regret(regrets = regrets, regrets_bp = regrets_bp,net = args.net, t=t, print_every=args.print_every,plot_vars=True,avg_vars=avg_vars, pick_vars_all=pick_vars_all)
                    plt.savefig('regrets.jpg')
                    plt.close()

        if t == args.T-1: #This way saves plot only in final form

            print(learner.unique_data['graph_indices'])
            print(learner.data['graph_indices'])
            print("Max Reward Index:", np.argmax(graph_rewards))
            print('Vars:', vars)
            plot_indices = np.arange(args.num_actions)

            plt.figure()
            print('plot')
            plt.plot(means, '-', label='means', color='#9dc0bc', linewidth=0.6)
            #plt.scatter(plot_indices, means, '-', label='means', color='#9dc0bc')
            plt.title(f'Confidence and mean Estimates, t = {t}')
            plt.fill_between(np.arange(args.num_actions), means - np.sqrt(args.exploration_coef) * vars,
                                        means + np.sqrt(args.exploration_coef) * vars, alpha=0.2, color='#b2edc5')
            plt.plot(graph_rewards, label='true function', color='#7c7287', linewidth=0.6)
            #plt.scatter(plot_indices, graph_rewards, label='true function', color='#7c7287')
            color = [item * 255 / (t + 1) for item in np.arange(t + 1)]
            plt.scatter(actions_all,
                                evaluate(idx_list=actions_all, noisy=False, reward_list=graph_rewards, noise_var=args.noise_var,
                                            _rds=env_rds), c=color, s=1/4)
            print(len(color))
            plt.set_cmap('magma')
            plt.legend()
            plt.savefig('meanandvariances.svg', format='svg')
            plt.close()
            command = 'svg42pdf meanandvariances.svg meanandvariances.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            #print('Final Means:',means)
            #print('Final upper confidences:', means + np.sqrt(args.exploration_coef) * vars)
            #plt.show()
            #loss = torch.nn.MSELoss()
            #print('MSELOSS:', loss(torch.tensor(means),torch.tensor(graph_rewards)))

            # exp_result_file = os.path.join("/local/bsoyuer/base_code/graph_BO/results/gnnucb_acquired_pts", '%s.json' % 'action_set')
            # with open(exp_result_file, 'w') as f:
            #     json.dump(graph_data, f, indent=4, cls=NumpyArrayEncoder)
            # print('Dumped results to %s' % exp_result_file)   

            # exp_result_file = os.path.join("/local/bsoyuer/base_code/graph_BO/results/gnnucb_acquired_pts", '%s.json' % 'acquired_set')
            # with open(exp_result_file, 'w') as f:
            #     json.dump(learner.data, f, indent=4, cls=NumpyArrayEncoder)
            # print('Dumped results to %s' % exp_result_file)

            MAX_NUM_NODES = 29
            NUM_ACTIONS = args.num_actions

            #print('Graph Data:', graph_data)
            collected_indices = learner.data["graph_indices"]
            #print('Collected indices:',collected_indices)

            index = list(np.arange(len(graph_data)))
            #print('Index:', index)

            def feat_pad(feat_mat):
                return torch.nn.functional.pad(feat_mat,pad=(0,0,0,MAX_NUM_NODES-len(feat_mat)), value=0)#value=float('nan'))

            features_list = []
            rewards_list = []

            for ix in index:

                features_list.append(feat_pad(torch.tensor(graph_data[ix].feat_mat)).flatten())
                rewards_list.append(graph_rewards[ix]) 

            rewards_arr = np.array(rewards_list)
            rewards_arr.resize((len(graph_data),1))

            #for e in features_list:
                #print(e.shape)
            
            features_list = torch.stack(features_list).numpy().astype(np.float32)

            #print("Features_list_shape:", features_list.shape)
            #print('Type Features List:', type(features_list))

            reducer = umap.UMAP(n_neighbors=int(NUM_ACTIONS/10), min_dist=0.2)
            embedding = reducer.fit_transform(features_list)

            def rand_jitter(arr):
                stdev = .005 * (max(arr) - min(arr))
                return arr + np.random.randn(len(arr)) * stdev


            fig, axs = plt.subplots(2, 1)

            axs[0].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c='black',
                s = 10)
            axs[0].title.set_text('UMAP projection of Action Set')

            collected_embeddings = embedding[collected_indices]

            pcm1 = axs[0].scatter(
                rand_jitter(collected_embeddings[:, 0]),
                rand_jitter(collected_embeddings[:, 1]),
                c=np.arange(0, collected_embeddings.shape[0]),
                cmap='gist_ncar',
                marker = "x",
                s = 8,
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
                s = 8)
            axs[1].title.set_text('Action Set Rewards',)
            axs[1].set_xlim([np.min(collected_embeddings[:, 0])-1, np.max(collected_embeddings[:,0])+1])
            axs[1].set_ylim([np.min(collected_embeddings[:, 1])-1, np.max(collected_embeddings[:, 1])+1])
            fig.colorbar(pcm2, ax=axs[1])

            fig.set_size_inches(18.5, 18.5)
            fig.savefig('collected_pts.svg', format='svg')
            plt.close()
            command = 'svg42pdf collected_pts.svg collected_pts.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards).flatten()), torch.max(torch.tensor(means).flatten())]))
            min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards).flatten()), torch.min(torch.tensor(means).flatten())]))
            #print(max_val)

            plt.figure()
            conf_bounds = np.sqrt(args.exploration_coef)*np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
            cmap = matplotlib.colors.ListedColormap(['red', 'green'])
            plt.errorbar(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), xerr=conf_bounds, fmt='o', alpha=0.2)
            plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), \
                        c=np.isin(np.arange(args.num_actions), learner.data['graph_indices']), cmap=cmap, s=1/2)
            plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
            plt.title('Seen samples vs Unseen Samples')
            plt.colorbar()
            plt.legend()
            plt.xlabel('Predicted')
            plt.ylabel("True Reward")

            plt.savefig('means_vs_rewards_seen_vs_unseen.svg', format='svg')
            plt.close()
            command = 'svg42pdf means_vs_rewards_seen_vs_unseen.svg means_vs_rewards_seen_vs_unseen.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            plt.figure()
            means_timestep = np.array([learner.get_post_mean(idx) for idx in actions_all])
            graph_rewards_timestep = evaluate(idx_list=actions_all, noisy=False, reward_list=graph_rewards, noise_var=args.noise_var,
                                            _rds=env_rds)
            colors_wrt_timesteps = [item * 255 / (t + 1) for item in np.arange(t + 1)]
            plt.scatter(torch.tensor(means_timestep).flatten(), torch.tensor(graph_rewards_timestep).flatten(), c=colors_wrt_timesteps, cmap="plasma", s=1/2)
            plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
            plt.title('Fit With Respect To Timesteps')
            plt.colorbar()
            plt.legend()
            plt.xlabel('Predicted')
            plt.ylabel("True Reward")

            plt.savefig('means_vs_rewards_wrt_timesteps.svg', format='svg')
            plt.close()
            command = 'svg42pdf means_vs_rewards_wrt_timesteps.svg means_vs_rewards_wrt_timesteps.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            

            plt.figure()
            plt.scatter(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), c=graph_rewards, cmap="plasma", s=1/2)
            plt.plot([min_val, max_val], [min_val, max_val], alpha=0.3)
            plt.title('Fit With Respect To Rewards')
            plt.colorbar()
            plt.legend()
            plt.xlabel('Predicted')
            plt.ylabel("True Reward")

            plt.savefig('means_vs_rewards_wrt_rewards.svg', format='svg')
            plt.close()
            command = 'svg42pdf means_vs_rewards_wrt_rewards.svg means_vs_rewards_wrt_rewards.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            
            plt.figure()
            plt.plot(seen_losses, '-', label='seen', color='#9dc0bc')
            plt.title('Losses per Steps (After #Pretraining Steps)')
            plt.plot(unseen_losses, label='unseen', color='#7c7287')
            plt.savefig('seen_and_unseen_losses.svg', format = 'svg')
            plt.close()
            command = 'svg42pdf seen_and_unseen_losses.svg seen_and_unseen_losses.pdf'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

    if args.runner_verbose:
        print(f'{learner.name} with {args.T} steps takes {(time.time() - t0)/60} mins.')
    exp_results = {'actions': actions_all, 'rewards': pick_rewards_all, 'regrets': regrets, 'regrets_bp': regrets_bp, 'pick_vars_all': pick_vars_all, 'avg_vars':avg_vars}

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
    parser.add_argument('--runner_verbose', type=bool, default=True)

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

    parser.add_argument('--data', type=str, default='QM9DATA', help='dataset type')
    #parser.add_argument('--synthetic', action='store_true') #If you dont specify synthetic, False, if you put only '--synthetic', True
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--dataset_size', type=int, default=130831)
    parser.add_argument('--num_actions', type=int, default=100, help = 'size of the actions set, i.e. total number of graphs')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help = 'number of MLP layer for the GNTK that creates the synthetic data')

    parser.add_argument('--stop_count', type=int, default=1000)
    parser.add_argument('--relative_improvement', type=float, default=1e-4)
    parser.add_argument('--small_loss', type=float, default=1e-3)

    parser.add_argument('--load_pretrained', default=False, action='store_true', help='Bool type')

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
    

    args = parser.parse_args()
    main(args)