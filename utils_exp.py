import sys
import os
import json
import time
import argparse

import numpy as np
import glob
import pandas as pd
from config import DATA_DIR, RESULT_DIR
from graph_env.graph_generator import Graph

""" Gather/READ synthetic data WHICH IS ALREADY GENERATED OR ALREADY AVAILABLE """

def collect_dataset(args, verbose=True):
    exp_dir = os.path.join(DATA_DIR, args.data)
    no_results_counter = 0
    data_dicts = []
    param_names = set()
    data_counter = 0
    #print(args.synthetic)
    for results_file in glob.glob(exp_dir + '/*.json'):
        #print('Loading Dataset #', data_counter+1)
        data_counter += 1
        if args.synthetic:
            if os.path.isfile(results_file):
                try:
                    with open(results_file, 'r') as f:
                        data_dict = json.load(f) #IS I THINK A 2D DICT: PARAMS--->NUMNODES,EDGEPROB, FEATDIM AND DATASET----->????
                    if (data_dict['params']['num_nodes'] == args.num_nodes) and (data_dict['params']['edge_prob'] == args.edge_prob) and (data_dict['params']['feat_dim'] == args.feat_dim):
                        data_dicts.append({**data_dict['dataset'], **data_dict['params']}) #NOTE: "**" EXTRACTS VALUES IN DICTS CORRESPONDING TO KEYS AND PASSES THEM TO THE FUNCTION APPEND() SO THAT
                        #DATASET AND PARAMS PART OF THE DICTIONARY FOR EACH DATAPT ARE PACKED AS A SINGLE DICT AND ADDED TO THE LIST OF DICTS
                        param_names = param_names.union(set(data_dict['params'].keys())) #Gather the union of keys under oarams of each considered datadict
                        #print('Dataset Matches description!')
                except json.decoder.JSONDecodeError as e:
                    print(f'Failed to load {results_file}', e)
            else:
                no_results_counter += 1
        else:
            if os.path.isfile(results_file):
                try:
                    with open(results_file, 'r') as f:
                        data_dict = json.load(f) #IS I THINK A 2D DICT: PARAMS--->NUMNODES,EDGEPROB, FEATDIM AND DATASET----->????
                    if (data_dict['params']['dataset_size'] == args.dataset_size) and (data_dict['params']['num_mlp_layers'] == args.num_mlp_layers):
                        data_dicts.append({**data_dict['dataset'], **data_dict['params']}) #NOTE: "**" EXTRACTS VALUES IN DICTS CORRESPONDING TO KEYS AND PASSES THEM TO THE FUNCTION APPEND() SO THAT
                        #DATASET AND PARAMS PART OF THE DICTIONARY FOR EACH DATAPT ARE PACKED AS A SINGLE DICT AND ADDED TO THE LIST OF DICTS
                        param_names = param_names.union(set(data_dict['params'].keys())) #Gather the union of keys under oarams of each considered datadict
                        #print('Dataset Matches description!')
                except json.decoder.JSONDecodeError as e:
                    print(f'Failed to load {results_file}', e)
            else:
                no_results_counter += 1


    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            args.data, len(data_dicts), no_results_counter))

    return pd.DataFrame(data=data_dicts), list(param_names) #Data frame with two objects, at 0th index: an array where datadicts are rows and columns are labelled by keys
    #and at  1st index, we have list of union of param names 

def dataset_to_graphdata(dataset, args):
    graph_features = list(dataset['features'])[0]
    graph_connections = list(dataset['connections'])[0]
    if args.synthetic:
        num_nodes = list(dataset['num_nodes'])[0]
    else:
        num_nodes = 0 #dummy value
    feat_dim = list(dataset['feat_dim'])[0] #Generate a list of graph data structs based on  features, adj matrices etc. that fully specify a graph
    graph_data = [Graph(dim_feats=feat_dim, num_nodes=num_nodes,adj_mat=np.array(adj_mat),feat_mat=np.array(feat_mat), synthetic=args.synthetic ) for adj_mat, feat_mat in zip(graph_connections,graph_features)]
    return graph_data #Here, if our features and connections are nonrandom, then simply returns
    #a graph struct with those non random data, no edge or feature sampling etc.

def read_dataset(args, env_rds):
    t0 = time.time()
    if args.synthetic:
        # Read the datasets
        datasets, _= collect_dataset(args)
        # pick ONE dataset that matches the environment setting, hence the below '==' lines (UNDER GRAPH_ENV' ENVIRONMENT.PY)
        #ONLY TAKE THE DATAPTS
        #FALLING UNDER THE PARAMETERS OF THE ENV SPECIFIEC FOR THE EXPERIMENT
        datasets = datasets.loc[datasets['num_nodes'] == args.num_nodes]
        datasets = datasets.loc[datasets['edge_prob'] == args.edge_prob]
        datasets = datasets.loc[datasets['feat_dim'] == args.feat_dim]
        datasets = datasets.loc[datasets['noise_var'] == args.noise_var]
        datasets = datasets.loc[datasets['num_mlp_layers'] == args.num_mlp_layers]
        env_seed = env_rds.choice(datasets['env_seed'])
        datasets = datasets.loc[datasets['env_seed'] == env_seed]
        graph_rewards = list(datasets['rewards'])[0]
        graph_rewards = [item for sublist in graph_rewards for item in sublist]
        graph_data = dataset_to_graphdata(datasets, args)
        print('Loading data took:', (time.time()-t0)/60)
        return graph_data, graph_rewards #FINALLY RETURNS THE GRAPH DATA AND DESIRED FORM
    else:
        datasets, _= collect_dataset(args)#Pick ONE dataset that matches the desired properties
        datasets = datasets.loc[datasets['dataset_size'] == args.dataset_size]
        #datasets = datasets.loc[datasets['num_mlp_layers'] == args.num_mlp_layers]
        env_seed = env_rds.choice(datasets['env_seed'])
        datasets = datasets.loc[datasets['env_seed'] == env_seed]
        graph_rewards = list(datasets['rewards'])[0]
        graph_rewards = [item for sublist in graph_rewards for item in sublist]
        graph_data = dataset_to_graphdata(datasets, args)
        print('Loading data took:', (time.time()-t0)/60)
        return graph_data, graph_rewards #FINALLY RETURNS THE GRAPH DATA AND DESIRED FORM



''' DIDNT READ AFTER HERE SINCE I BELIEVE IT IS EULER AND OR GATHERING EXPERIMNET RESULTS RELATED'''

""" Gather exp results """

def collect_exp_results(exp_name, verbose=True):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    no_results_counter = 0
    print(exp_dir)
    exp_dicts = []
    param_names = set()
    for results_file in glob.glob(exp_dir + '/*/*.json'): #might have to change the regex thing
        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                try:
                    exp_dicts.append({**exp_dict['exp_results'], **exp_dict['params'], **{'algorithm': exp_dict['algorithm']},  **{'time': exp_dict['time']}})
                except:
                    exp_dicts.append({**exp_dict['exp_results'], **exp_dict['params'], **{'algorithm': exp_dict['algorithm']}})
                param_names = param_names.union(set(exp_dict['params'].keys()))
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    if verbose:
        print('Parsed results %s - found %i folders with results and %i folders without results' % (
            exp_name, len(exp_dicts), no_results_counter))

    return pd.DataFrame(data=exp_dicts), list(param_names)

""" Async executer """
import multiprocessing

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass


""" Command generators """

def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd

def cmd_exec_fn(cmd):
    import os
    os.system(cmd)

def generate_run_commands(command_list, num_cpus=1, dry=False, n_hosts=1, mem=10000, long=False,
                          mode='local', promt=True, log_file_list=None):

    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'sbatch ' + \
                   f'--time={23 if long else 3}:59:00 ' + \
                   f'--ntasks={num_cpus} ' + \
                   f'--mem-per-cpu={mem} '
                   #f"span[hosts={n_hosts}]"

        if log_file_list is not None:
            assert len(command_list) == len(log_file_list)

        for python_cmd in command_list:
            print('Python:',python_cmd)
            if log_file_list is not None:
                log_file = log_file_list.pop()
                cluster_cmds.append(bsub_cmd + f'-o {log_file} -e {log_file} ' + ' --wrap=' + '"' + python_cmd + '"')
            else:
                cluster_cmds.append(bsub_cmd + ' --wrap=' + '"' + python_cmd + '"')

        if promt:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the euler cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local':
        if promt:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(f"About to launch {len(command_list)} commands in {num_cpus} local processes. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                exec.run(cmd_exec_fn, command_list)
    else:
        raise NotImplementedError

""" Hashing and Encoding dicts to JSON """

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def hash_dict(d):
    return str(abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))

if __name__ == '__main__':
    DIR = '/local/bsoyuer/base_code/graphBO/'
    #df_full, _ = collect_exp_results(exp_name='testing_pipeline/NN')
    parser = argparse.ArgumentParser(description='Utils_exp Run')
    parser.add_argument('--num_nodes', type=int, default=100, help = 'max number of nodes per graph')
    parser.add_argument('--feat_dim', type = int, default=100, help ='Dimension of node features for the graph')
    parser.add_argument('--edge_prob', type=float, default=0.2, help='probability of existence of each edge, shows sparsity of the graph')
    parser.add_argument('--data_size', type=int, default=5, help = 'size of the seed dataset for generating the reward function')
    parser.add_argument('--num_actions', type=int, default=200, help = 'size of the actions set, i.e. total number of graphs')
    parser.add_argument('--noise_var', type=float, default=0.0001, help = 'variance of noise for observing the reward, if exists')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help = 'number of MLP layer for the GNTK that creates the synthetic data')
    parser.add_argument('--seed', type=int, default=354)
    parser.add_argument('--nn_init_lazy', type=bool, default=True)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--print_every', type=str, default=20)
    parser.add_argument('--runner_verbose', type=bool, default=True)


    #parser.add_argument('--data', type=str, default='QM9 DATA', help='dataset type') #Set to synthetic if need be
    #parser.add_argument('--synthetic', type=bool, default=False) #Used here and in graph_env/graph_generator.py
    #basically controls how the graph struct is generated when num_nodes isn't fixed in real datasets.
    #parser.add_argument('--dataset_size', type=int, default=130831)
    #To select ONE dataset of size data_size when its a real life dataset

    args = parser.parse_args()
    env_rds = np.random.RandomState(args.seed)

    data, rewards = read_dataset(args, env_rds)

    #print(rewards)
    print(len(rewards))
    print(data[57].adj_mat)