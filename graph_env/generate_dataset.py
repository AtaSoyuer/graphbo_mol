import sys 
import os
sys.path.append(os.path.abspath("/cluster/scratch/bsoyuer/base_code/graph_BO/graph_env"))
from environment import Reward
from graph_generator import generate_graph_data
import time
import numpy as np
import json
sys.path.append('/cluster/scratch/bsoyuer/base_code/graph_BO')
from config import  *
from utils_exp import NumpyArrayEncoder
import ray
import argparse

'''
I THINK THIS IS THE ULTIMATE SCRIPT FOR SYNTHETIC DATA GENERATION. USING THE REWARD GENERATING
ENVIRONMENT SCRIPT USING KERNEL GENERATOR AND THE GRAPH GENERATOR SCRIPT, THIS SCRIPT CREATED
SYNTHETIC RANDOM GRAPH DATA WITH REWARDS SAMPLED FROM GNTK POSTERIR UNDER SOME NOISE
NOTE: IF YOU GET ERROR 0xE 0xD ert RUNNING THIS SCRIPT, PIP UNINSTALL NUMPY (MAYBE TWICE)
AND RE-INSTALL VERSION 1.21.1!!!!
'''

# args = {
#     'data': 'synthetic_data',
#     'env_seed': 865,
#     'num_batches': 1000,
#     'data_size': 5,
#     'num_nodes': 5, #N = 5, 20, 100
#     'edge_prob': 0.2, #p = 0.05, 0.2, 0.95
#     'feat_dim': 10  #d = 10, 100, 500
# }

def main(args):
    print("Data dir is:",DATA_DIR)
    print("Base dir is:",BASE_DIR)
    #set path
    exp_result_folder = os.path.join(DATA_DIR, args.data)

    # set the environment and the base variables
    sampling_mode = 'binomial'
    noise_var = 0.0001
    #noise_var = 0.01
    num_mlp_layers = 2
    batch_size = 20
    num_actions = args.num_batches * batch_size
    data_config = {'env_seed': args.env_seed, 'seeddata_size': args.data_size, 'num_nodes': args.num_nodes,
                     'feat_dim': args.feat_dim, 'num_mlp_layers': num_mlp_layers, 'noise_var': noise_var, 'edge_prob': args.edge_prob}
    print(f'N={args.num_nodes} , d={args.feat_dim}, p={args.edge_prob}, |X|={num_actions}')

    ray.init(num_cpus = args.num_cpus)
    @ray.remote #Allows completing the below method in a parallelized sense over the cpus
    def generate_batched_reward(graph_batch:list, env_seed:int, seeddata_size, num_nodes, feat_dim, num_mlp_layers, noise_var, edge_prob):
        env_rds = np.random.RandomState(env_seed)
        reward = Reward(evidence_size=seeddata_size, num_nodes=num_nodes, dim_feats=feat_dim, num_mlp_layers=num_mlp_layers,
                        noise_var=noise_var, edge_prob=edge_prob, random_state=env_rds)
        graph_rewards = reward.rew_func(graph_batch)
        return graph_rewards


    #sample graphs
    t0 = time.time()
    env_rds = np.random.RandomState(args.env_seed)
    graph_data = generate_graph_data(data_size=num_actions, num_nodes=args.num_nodes, dim_feats=args.feat_dim,
                                     sampling_mode=sampling_mode, edge_prob=args.edge_prob, random_state=env_rds)
    graph_data_features = [graph.feat_mat for graph in graph_data]
    graph_data_connections = [graph.adj_mat for graph in graph_data]
    #All 3 graph data above are lists
    print('time for sampling graphs:', (time.time() - t0) / 60)
    t0 = time.time()

    # # initialize Reward class
    # env_rds = np.random.RandomState(args.env_seed)
    # reward = Reward(evidence_size=args.data_size, num_nodes=args.num_nodes, dim_feats=args.feat_dim, num_mlp_layers=num_mlp_layers,
    #                 noise_var=noise_var, edge_prob=args.edge_prob, random_state=env_rds)

    # cut into batches - slice graph_data by batch_size steps and take each slice/batch as list element in batches_graph
    batched_graph_data = [graph_data[batch_size * i: batch_size * (i + 1)] for i in range(args.num_batches)]

    # calculate rewards for each batch- conducted in parallel via above function
    graph_rewards_parallel = [generate_batched_reward.remote(graph_batch = graph_batch,**data_config) for graph_batch in batched_graph_data]
    graph_rewards = np.concatenate(ray.get(graph_rewards_parallel))
    #Here, need to call ray.get to get actual results outputted by executing graph_rewards_parallel
    #instead of getting just the object identifier objects. The returned list of rewards are concatenated in a numpy array

    dataset = {'features': graph_data_features, 'connections': graph_data_connections, 'rewards': graph_rewards}
    '''
    This is the dataset format I believe as a dictionary!!!
    '''

    total_time = (time.time() - t0) / 60
    print('calculating reward for the new graphs:', (time.time()-t0)/60)

    data_config['sampling_mode'] = sampling_mode
    data_config['dataset_size'] = num_actions

    # make a dictionary of the results

    """ store results """
    results_dict = {
        'params': data_config,
        'dataset': dataset,
        'duration_total': total_time
    }

    if exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict) #A fancy print to print the results dictionary in nice form
    else:
        os.makedirs(exp_result_folder, exist_ok=True)
        # hash this to create a name, i.e. as you see 2 lines below, hashing assings a
        #unique name to each data object which we use infront of .json when saving
        data_hash = str(abs(json.dumps(results_dict['params'], sort_keys=True).__hash__()))
        exp_result_file = os.path.join(exp_result_folder, '%s.json' % data_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s' % exp_result_file) #Dump all datapts in results_dicts store
        #as dictionaries to a json file with given hashes.
        print(f'N={args.num_nodes} , d={args.feat_dim}, p={args.edge_prob}, |X|={num_actions}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Generator')

    # environment arguments
    # this is to set which dataset to pick
    parser.add_argument('--data', type=str, default='synthetic_data', help='dataset type')
    parser.add_argument('--env_seed', type=int, default=875, help='random seed')
    parser.add_argument('--num_nodes', type=int, default=80, help = 'max number of nodes per graph')
    parser.add_argument('--feat_dim', type = int, default=100, help ='Dimension of node features for the graph')
    parser.add_argument('--edge_prob', type=float, default=0.2, help='probability of existence of each edge, shows sparsity of the graph')
    parser.add_argument('--data_size', type=int, default=5, help = 'size of the seed dataset for generating the reward function')
    parser.add_argument('--num_batches', type=int, default=250, help = 'size of the actions set, i.e. total number of graphs would be num_batches*10')
    parser.add_argument('--num_cpus', type=int, default=32)
    parser.add_argument('--seed', type=int, default=796)
    args = parser.parse_args()
    main(args)