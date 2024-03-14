from utils_exp import hash_dict, AsyncExecutor
from config import RESULT_DIR
import run_gnnucb_pyg
import argparse
import numpy as np
import copy
import os
import itertools
import sys

def generate_base_command(module, boolean_args, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable #MAY NEED TO MANAULLY SET TO /cluster/project/krause/bsoyuer/miniconda3/envs/graphbo/bin/python
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    base_cmd += boolean_args
    return base_cmd

def cmd_exec_fn(cmd):
    import os
    os.system(cmd)

def generate_run_commands(command_list, num_cpus=1, dry=False, n_hosts=1, mem=10000, long=False,
                          mode='local', promt=True, gpu='titan_rtx:1', log_file_list=None):

    if mode == 'euler':
        cluster_cmds = []
        bsub_cmd = 'sbatch ' + \
                   '-A ls_krausea ' + \
                   f'--time={23 if long else 3}:59:00 ' + \
                   f'--gpus={gpu} ' + \
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


'''
WHEN ADDING NEW HYPERPARAMS TO SEARCH FOR, MAKE NECESSAY MOFICICATIONS IN LINES ~13, 32, 47!!!
ALSO, CHANGE NUM_CPUS, NUM DIFFERENT SEEDS PER HYPERPARAM CONFIGURATION (5) OR TOTAL NUM OF DIFFERENT HYPERPARAM
CONFIGURATIONS (25) IN BELOW PARSERS AT THE BOTTOM
CHANGE ARGS.EXP_NAME IN PARSESR BELOW TO SAVE JSONS TO A NEW DIRECTORY
'''

applicable_configs = {
    #'GNN-UCB':['exploration_coef','pretrain_steps', 'alg_lambda','neuron_per_layer','net', 't_intersect', 'neuron_per_layer', 'lr', 'stop_count', 'relative_improvement', 'small_loss', 'num_mlp_layers_alg'],
    'GNN_UCB': ['GD_batch_size','T','T0', 'T1', 'T2','alg_lambda','alpha','batch_size','batch_window_size','dim','dropout_prob','exploration_coef','explore_threshold','factor','gamma','net','lr', \
    'neuron_per_layer','num_mlp_layers_alg','num_actions','pretrain_steps','pool_num','print_every','patience','stop_count','small_loss','subsample_method', 'subsample_num', 'synthetic', 'relative_improvement', 'reward', \
    'pretrain_model_name', 'pretrain_indices_name'],
    'Dataset': ['num_nodes','feat_dim','edge_prob', 'num_actions']
}

default_configs = {
    # Dataset
    'num_nodes': 5, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 12, # 10 or 100 #CHANGE WHEN SYNTHETIC!!!!
    'num_actions': 126000, # any number below 10000 works.
    # GNN-UCB
    'GD_batch_size':50,
    'T':1251,
    'T0':80, 
    'T1':50, 
    'T2':400,
    'alg_lambda': 0.003,
    'alpha':2.5,
    'batch_size':50,
    'batch_window_size':80,
    'dim':64,
    'dropout_prob':0.2,
    'exploration_coef': 5.0,
    'explore_threshold': 10,
    'factor':0.7,
    'gamma':2.0,
    'net': 'GNN',
    'lr': 1e-3,
    'neuron_per_layer':128,
    'num_mlp_layers_alg': 1,
    'pretrain_steps': 100,
    'pretrain_model_name': 'pretrain_indices',
    'pretrain_indices_name': 'nnconv_reward0_8000samples_100ep',
    'pool_num':200,
    'print_every':50,
    'patience':5,
    'stop_count' : 9000,
    'small_loss' : 1e-3,
    'subsample_method':'random', 
    'subsample_num':20, 
    'synthetic':0,
    'relative_improvement' : 1e-6,
    'reward':0
}

boolean_args = ' --alternative' + ' --batch_GD' + ' --pool' + ' --load_pretrained' + ' --focal_loss' + ' --large_scale '

# search_ranges = {
#     # #GNN-UCB
#     # 'exploration_coef': ['loguniform',[-5,0]],
#     # 'alg_lambda': ['loguniform', [-5,-1]], #keep it small.
#     # GNN-US
#     'exploration_coef': ['loguniform', [-2, 0]],
#     'alg_lambda': ['loguniform', [-4, -3]],  # keep it small.
#     'pretrain_steps': ['intuniform', [60,100]],
#     #'t_intersect': ['intuniform', [100,600]],
#     'neuron_per_layer': ['specified', [4096,8192]],
#     'lr':['loguniform', [-5, -3]],
#     'stop_count' : ['endpoints_5', [4000, 6000]],
#     'relative_improvement' : ['loguniform', [-7, -5]],
#     'small_loss' : ['loguniform', [-4, -2]],
#     'num_mlp_layers_alg': ['single', [1]]
# }


# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys()}

def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10**rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    elif sample_type == 'intuniform':
        return rds.randint(*range)
    elif sample_type == 'specified':
        return rds.choice(np.array([range[0], range[1]]))
    elif sample_type == 'endpoints_5':
        return rds.choice(np.linspace(range[0], range[1], num=5).astype('int'))
    elif sample_type == 'endpoints_2':
        return rds.choice(np.linspace(range[0], range[1], num=2).astype('int'))
    elif sample_type == 'single':
        return range[0]
    else:
        raise NotImplementedError

def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds < 101
    init_seeds = list(rds.randint(0, 10**6, size=(101,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    #exp_path = os.path.join(exp_base_path, '%s'%(args.net))
    exp_path = exp_base_path


    command_list = []
    
    # transfer flags from the args
    flags = copy.deepcopy(args.__dict__)
    [flags.pop(key) for key in ['seed', 'num_seeds', 'exp_name', 'num_cpus', 'mode', 'gpu', 'mem']] #REMOVE THESE KEYS FROM DICT OF ARGS AS THE REMAINDER WILL BE USED TO GENERATE BASE COMMAND

    # randomly sample flags
    for flag in default_configs:
        #if flag in search_ranges:
            #flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
        #else:
        flags[flag] = default_configs[flag]

    # determine subdir which holds the repetitions of the exp
    flags_hash = hash_dict(flags)
    flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)
    print(flags)
    for j in range(args.num_seeds):
        seed = init_seeds[j]
        #cmd = generate_base_command(run_phasedgp, flags=dict(**flags, **{'seed': seed}))
        cmd = generate_base_command(run_gnnucb_pyg, boolean_args=boolean_args, flags=dict(**flags, **{'seed': seed}))
        command_list.append(cmd)

    # submit jobs
    #generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode=args.mode,long=True, promt=True, mem=args.mem, gpu=args.gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--exp_name', type=str,  default='hyperparamgnnucb_bartu_10',  help='subdir in results/ to save the result dicts to')
    parser.add_argument('--num_cpus', type=int, default=16)
    #parser.add_argument('--num_hparam_samples', type=int, default=40)
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--data', type=str, default='QM9DATA', help='dataset type')
    parser.add_argument('--mode', type=str, default='euler', help='euler, local etc. for generate run command')
    parser.add_argument('--gpu', type=str, default='titan_rtx:1', help='gpu to use on euler')
    parser.add_argument('--mem', type=int, default=5000, help='euler, local etc. for generate run command')
    parser.add_argument('--seed', type=int, default=864, help='random number generator seed')
    #parser.add_argument('--runner_verbose', type=bool, default=False)

    # model arguments
    # this is to set algo params that you don't often want to change
    #parser.add_argument('--net', type=str, default='GNN', help='Network to use for UCB')
    #parser.add_argument('--nn_aggr_feat', type=bool, default=True)
    #parser.add_argument('--nn_init_lazy', type=bool, default=True)

    #parser.add_argument('--batch_size', type=int, default=20)
    #parser.add_argument('--T', type=int, default=500) #change to 1500
    #parser.add_argument('--T0', type=int, default=150)
    parser.add_argument('--dataset_size', type=int, default=130831)
    #parser.add_argument('--synthetic', type=int, default=0)

    ''' CHANG THESE PARAMETERS AND FEAT_DIM ABOVE WHEN UYSING SYNTHETIC DATA'''

    args = parser.parse_args()
    main(args)