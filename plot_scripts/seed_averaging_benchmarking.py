from utils_exp import hash_dict, AsyncExecutor
from config import RESULT_DIR
import run_gnnucb_pyg, run_gnnucb_pyg_mp, run_gpucb_mp
import argparse
import numpy as np
import copy
import os
import itertools
import sys

def generate_base_command(boolean_args, runner = 'run_gnnucb_pyg', flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable #MAY NEED TO MANAULLY SET TO /cluster/project/krause/bsoyuer/miniconda3/envs/graphbo/bin/python
    if runner == 'run_gnnucb_pyg':
        base_exp_script = os.path.abspath(run_gnnucb_pyg.__file__)
    elif runner == 'run_gnnucb_pyg_mp':
        base_exp_script = os.path.abspath(run_gnnucb_pyg_mp.__file__)
    elif runner == 'run_gpucb_mp':
        base_exp_script = os.path.abspath(run_gpucb_mp.__file__)
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
WHEN ADDING NEW HYPERPARAMS, ADD THEM TO APPLICABLE_CONFIGS, DEFAULT_CONFIGS. IF YOU ARE GOING TO SEARCH OVER THEM, ADD THEM TO
SEARCH_RANGES AS WELL. NOTE: BOOLEAN HYPERPARAMS VALUES MUST BE SET TO TRUE OR FALSE IN DEFAULT CONFIGS AND ONLY THEY
CAN HAVE BOOLEAN VALUES TO DIFFERENTIATE THEM FROM OTHER HYPERPARAMS DURING RUN_CMD!!!!! 

ALSO, WHEN TESTING FOR A NEW BOOLEAN FLAG
IN SEARCH RANGES, IF IT HAS ANOTHER ASSOCIATED HYPERPARAMETER (EX:BATCH_WINDOW AND BATCH_WINDOW_SIZE), ADD a CONDITION JUST AFTER
THE MAIN LOOP STARTS SO THAT FOR ALL CONFIGS WHERE THE BOOLEAN FLAG IS FALSE, ITS ASSOCIATED HYPERPARAM HAS ITS VALUES SET TO THE
VALUE IN DEFAULT CONFIGS IN ALL CONFIGURATIONS WHICH ALLOWS LATER IN THE PLOTTING SCRIPT, TO GROUP ALL THE EXPERIMENTS WHERE BOOLEAN
IS FALSE TOGETHER. NOTE: FOR THIS TO WORK, NEED TO HAVE BOOLEAN HYPERPARAMS COMING IN FIRST AT DEFAULT_CONFIGS!!!! AND JUST TO MAKE SURE
SEARCH RANGES DOESNT YIELD RANDOMLY THE VALUE THE SAME AS IN DEFAULT CONFIGS, SET THE VALUE OF THE ASSOCIATED HYPERPARAM IN DEFAULT CONFIGS
TO AN ARBITRARY VALUE OUTSIDE OF THE POSSIBLE SEARCH RANGE!!!

ALSO, CHANGE NUM_CPUS, NUM DIFFERENT SEEDS PER HYPERPARAM CONFIGURATION (5) OR TOTAL NUM OF DIFFERENT HYPERPARAM
CONFIGURATIONS (25) IN BELOW PARSERS AT THE BOTTOM
CHANGE ARGS.EXP_NAME IN PARSESR BELOW TO SAVE JSONS TO A NEW DIRECTORY
'''

applicable_configs = {
    #'GNN-UCB':['exploration_coef','pretrain_steps', 'alg_lambda','neuron_per_layer','net', 't_intersect', 'neuron_per_layer', 'lr', 'stop_count', 'relative_improvement', 'small_loss', 'num_mlp_layers_alg'],
    'GNN_UCB': ['GD_batch_size','T','T0', 'T1', 'T2','alg_lambda','alpha','batch_size','batch_window_size','dim','dropout_prob','exploration_coef','explore_threshold','factor','gamma','net','lr', \
    'neuron_per_layer','num_mlp_layers_alg','num_actions','pretrain_steps','pool_num','print_every','patience','stop_count','small_loss','subsample_method', 'subsample_num', 'synthetic', 'relative_improvement', 'reward', \
    'pretrain_model_name', 'pretrain_indices_name', 'reward_plot_dir', 'alternative', 'batch_GD', 'pool', 'load_pretrained', 'focal_loss', 'large_scale', 'pool_top_means', 'ucb_wo_replacement', 'batch_window', 'batch_window_size', #'num_epochs',
    'small_net_var', 'initgrads_on_fly', 'no_var_computation', 'oracle', 'select_K_together', 'select_K', 'laplacian_features', 'laplacian_k', 'pretraining_load_pretrained', 'pretraining_pretrain_model_name', 'pretrain_num_indices', 'remove_smiles_for_sober', 'runner_verbose', 'thompson_sampling', 'rand'],
    'Dataset': ['num_nodes','feat_dim','edge_prob', 'num_actions']
}

default_configs = {
    # Dataset
    'num_nodes': 5, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 12, # 10 or 100 #CHANGE WHEN SYNTHETIC!!!!
    'num_actions': 130831, # any number below 10000 works.
    #BOOLEAN ARGS:
    'alternative':'true',
    'batch_GD':'true',
    'pool':'true',
    'load_pretrained':'true',
    'large_scale':'false',
    'ucb_wo_replacement':'true',
    'focal_loss':'false',
    'pool_top_means':'false',
    'batch_window':'false',
    'small_net_var': 'true',
    'initgrads_on_fly': 'false',
    'no_var_computation': 'false',
    'oracle': 'false',
    'select_K_together': 'false',
    'laplacian_features': 'false',
    'pretraining_load_pretrained': 'false',
    'remove_smiles_for_sober': 'true',
    'runner_verbose': 'false',
    'thompson_sampling': 'true',
    'rand': 'false',
    # GNN-UCB
    'GD_batch_size':50,
    'T':2001,
    'T0':80, 
    'T1':50, 
    'T2':100,
    'alg_lambda': 0.003,
    'alpha':2.5,
    'batch_size':100,
    'batch_window_size':10,
    'dim':64,
    'dropout_prob':0.2,
    'exploration_coef': 0.5,
    'explore_threshold': 10,
    'factor':0.7,
    'gamma':2.0,
    'net': 'GNN',
    'lr': 1e-3,
    'laplacian_k': 1,
    'neuron_per_layer':128,
    'num_mlp_layers_alg': 1,
    #'num_epochs': 50,
    'pretrain_steps': 100,
    'pretrain_num_indices': 2000,
    'pretrain_model_name': 'nnconv_reward3_4000samples_100ep',
    'pretrain_indices_name': 'pretrain_indices_rew3',
    'pretraining_pretrain_model_name': 'nnconv_reward3n4_8000samples_100ep',
    'pool_num':600,
    'print_every':400,
    'patience':5,
    'select_K': 10,
    'stop_count' : 9000,
    'small_loss' : 1e-4,
    'subsample_method':'random', 
    'subsample_num':20, 
    'synthetic':0,
    'relative_improvement' : 1e-8,
    'reward':3,
    'reward_plot_dir':3,
}

search_ranges = {
    #'batch_size': ['endpoints_5_det', [50, 250]]
    #'exploration_coef': ['loguniform', [-1, 1]],
    #'alg_lambda': ['loguniform', [-4, -3]],  # keep it small.
    #'pretrain_steps': ['intuniform', [60,100]],
    #'neuron_per_layer': ['specified', [4096,8192]],
    #'lr':['loguniform', [-5, -3]],
    #'stop_count' : ['endpoints_5', [4000, 6000]],
    #'relative_improvement' : ['loguniform', [-7, -5]],
    #'small_loss' : ['loguniform', [-4, -2]],
    #'num_mlp_layers_alg': ['single', [1]],
    #'alternative' : ['specified', ['true', 'false']],
    #'batch_GD' : ['specified', ['true', 'false']],
    #'pool' : ['specified', ['true', 'false']],
    #'load_pretrained' : ['specified', ['true', 'false']],
    #'focal_loss' : ['specified', ['true', 'false']],
    #'large_scale' : ['specified', ['true', 'false']],
    #'pool_top_means' : ['specified', [''true'', 'false']],
    #'ucb_wo_replacement' : ['specified', ['true', 'false']],
    #'batch_window': ['weighted_boolean', ['true', 'false']],
    #'batch_window_size': ['intuniform', [400,1200]],
    #'batch_size': ['intuniform', [100,500]],
    #'pool_num': ['intuniform', [100,500]],
    #'T2': ['specified', [400, 2001]]
    }


# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys()}

def sample_flag(sample_spec, det_count = 0, rds=None):
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
    elif sample_type == 'weighted_boolean':
        return rds.choice(np.array([range[0], range[1]]), p=[0.75,0.25]) #Higher prob to set the boolean hyperparameter to True to 'Balance OUT' as described in annotation above
    elif sample_type == 'endpoints_2':
        return rds.choice(np.linspace(range[0], range[1], num=2).astype('int'))
    elif sample_type == 'endpoints_5_det':
        return np.linspace(range[0], range[1], num=5).astype('int')[det_count]
    elif sample_type == 'single':
        return range[0]
    else:
        raise NotImplementedError

def main(args):

    boolean_args = ' ' 

    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 101
    init_seeds = list(rds.randint(0, 10**6, size=(101,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    #exp_path = os.path.join(exp_base_path, '%s'%(args.net))
    exp_path = exp_base_path


    command_list = []
    det_count = 0
    batch_window_size_reminder = False
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'exp_name', 'num_cpus', 'mode', 'gpu', 'mem', 'num_hparam_samples', 'num_seeds_per_hparam', 'runner', 'deterministic_hparams']] #REMOVE THESE KEYS FROM DICT OF ARGS AS THE REMAINDER WILL BE USED TO GENERATE BASE COMMAND

        # randomly sample flags
        for flag in default_configs:
            if default_configs[flag] in ['true', 'false']: #Check if the current flag is a boolean one
                if flag in search_ranges:
                    flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
                else:
                    flags[flag] = default_configs[flag]

                if flag == 'batch_window':
                    if flags[flag] == 'false':
                        print('Batch_window=False Detected!')
                        batch_window_size_reminder = True


                if flags[flag] == 'true':
                    boolean_args = boolean_args + ' --' + flag + ' '

            else:
                if flag == 'batch_window_size':
                    if batch_window_size_reminder == True:
                        flags[flag] = default_configs[flag]
                        batch_window_size_reminder = False
                        print(f'set {flag} to {default_configs[flag]}')
                    else:
                        flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
                        print(f'set {flag} to {flags[flag]}')
                else:
                    if flag in search_ranges:
                        if args.deterministic_hparams:
                            flags[flag] = sample_flag(sample_spec=search_ranges[flag], det_count=det_count, rds=rds)
                            det_count += 1
                        else:  
                            flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
                    else:
                        flags[flag] = default_configs[flag]


        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)
        non_boolean_flags = {key:val for key, val in flags.items() if val != 'true'}
        non_boolean_flags = {key:val for key, val in non_boolean_flags.items() if val != 'false'}
        print('Flags:', non_boolean_flags)
        print('Boolean args:', boolean_args)
        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            #cmd = generate_base_command(run_phasedgp, flags=dict(**flags, **{'seed': seed}))
            cmd = generate_base_command(boolean_args=boolean_args, runner = args.runner, flags=dict(**non_boolean_flags, **{'seed': seed}))
            command_list.append(cmd)

        boolean_args = ''

    # submit jobs
    #generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode=args.mode,long=True, promt=True, mem=args.mem, gpu=args.gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--exp_name', type=str,  default='hyperparamgnnucb_bartu_10',  help='subdir in results/ to save the result dicts to')
    parser.add_argument('--num_cpus', type=int, default=16)
    #parser.add_argument('--num_hparam_samples', type=int, default=40)
    parser.add_argument('--num_hparam_samples', type=int, default=9, help='How many different hyperparameter configs in total')
    parser.add_argument('--num_seeds_per_hparam', type=int, default=3, help='How many different seeds per config, so results in num_hparam_samples*num_seeds_per_hparam many exps in total')
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--data', type=str, default='QM9DATA', help='dataset type')
    parser.add_argument('--mode', type=str, default='euler', help='euler, local etc. for generate run command')
    parser.add_argument('--gpu', type=str, default='titan_rtx:1', help='gpu to use on euler')
    parser.add_argument('--mem', type=int, default=5000, help='euler, local etc. for generate run command')
    parser.add_argument('--seed', type=int, default=864, help='random number generator seed')
    parser.add_argument('--runner', type=str, default='run_gnnucb_pyg', help='which algorithm runner to run')
    #parser.add_argument('--runner_verbose', type=bool, default=False)

    parser.add_argument('--dataset_size', type=int, default=130831)
    #parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--deterministic_hparams', default=False, action='store_true', help='Whether to choose hyperparam values to search over deteereministically from linsapce')

    ''' CHANG THESE PARAMETERS AND FEAT_DIM ABOVE WHEN UYSING SYNTHETIC DATA'''

    args = parser.parse_args()
    main(args)