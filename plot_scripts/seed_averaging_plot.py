import os
import sys
sys.path.append('/cluster/scratch/bsoyuer/base_code/graph_BO')
from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager
from plot_specs import *
import bundles
import argparse
import pandas as pd
import glob
import json
import pickle
#DIR = '/local/pkassraie/gnnucb/results/'

'''
CHANGE EXP_NAME IN THE ABOVE LINE TO READ FROM DIFFERENT DIRECTORY
WHEN READING DATA WITH NEW PARAMS, MAKE CHANGES IN LINES ~62, 73, 77, 79
ALSO, CAN READJUST GENERIC_LINES IN PLOT_SPECS.PY FOR BETTER LINE COLORS.
'''

# RESULT_DIR = '/cluster/scratch/bsoyuer/base_code/graph_BO/'

# def collect_exp_results(exp_name, verbose=True):
#     exp_dir = os.path.join(RESULT_DIR, exp_name)
#     no_results_counter = 0
#     print(exp_dir)
#     exp_dicts = []
#     param_names = set()
#     for results_file in glob.glob(exp_dir + '/*/*.json'): #might have to change the regex thing
#         if os.path.isfile(results_file):
#             try:
#                 with open(results_file, 'r') as f:
#                     exp_dict = json.load(f)
#                 exp_dicts.append({**exp_dict['exp_results'], **exp_dict['params'], **{'algorithm': exp_dict['algorithm']}})
#                 param_names = param_names.union(set(exp_dict['params'].keys()))
#             except json.decoder.JSONDecodeError as e:
#                 print(f'Failed to load {results_file}', e)
#         else:
#             no_results_counter += 1

#     if verbose:
#         print('Parsed results %s - found %i folders with results and %i folders without results' % (
#             exp_name, len(exp_dicts), no_results_counter))

#     return pd.DataFrame(data=exp_dicts), list(param_names)

configs = {
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
    'reward': 0
}


def main(args):
    top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]
    DIR = args.base_dir
    df_full, _ = collect_exp_results(exp_name=args.exp_dir)

    #Pick which QM9
    #df = df_full.loc[df_full['num_actions']==configs['num_actions']]
    df = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
    #df = df.loc[df['T'] == configs['T']]
    #print(df)


    #plot_name = 'new_paper_hyperparam_{}T_{}actions_QM9'.format(configs['T'], configs['num_actions'])
    plot_name = args.plot_name
    plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    fig_reg, axes_reg = plt.subplots( ncols = 1, nrows=1)#, figsize = (8, 12)
    plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    fig_topk, axes_topk = plt.subplots( ncols = 1, nrows=1)

    row_counter = 0
    col_counter =0
    for net in ['GNN']:
        counter = 0
        df_net = df.loc[df['net'] == net]
        #print(df_net)
        configurations = [config for config in
                        zip(df_net['alg_lambda'], 
                            df_net['exploration_coef'], 
                            df_net['pretrain_steps'], 
                            df_net['neuron_per_layer'], 
                            df_net['lr'], 
                            df_net['stop_count'], 
                            df_net['T'], 
                            df_net['small_loss'], 
                            df_net['reward'], 
                            df_net['GD_batch_size'],
                            df_net['T2'], 
                            df_net['dim'], 
                            df_net['gamma'], 
                            df_net['alpha'], 
                            df_net['pool_num'], 
                            df_net['batch_size'], 
                            df_net['num_actions']
                            ) if
                        not all(z == config[0] for z in config[1:])]
        configurations = list(set(configurations))
        # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
        for config in configurations:
            print(len(configurations))
            # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
            alg_lambda = config[0]
            exp_coef = config[1]
            pretrain_steps = config[2]
            neuron_per_layer = config[3]
            lr = config[4]
            stop_count = config[5]
            T = config[6]
            small_loss = config[7]
            reward = config[8]
            GD_batch_size = config[9]
            T2 = config[10]
            dim = config[11]
            gamma = config[12]
            alpha = config[13]
            pool_num = config[14]
            batch_size = config[15]
            num_actions = config[16]



            sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
            sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
            sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]
            sub_df = sub_df.loc[sub_df['neuron_per_layer'] == neuron_per_layer]
            sub_df = sub_df.loc[sub_df['lr'] == lr] 
            sub_df = sub_df.loc[sub_df['stop_count'] == stop_count] 
            sub_df = sub_df.loc[sub_df['T'] == T] 
            sub_df = sub_df.loc[sub_df['small_loss'] == small_loss]
            sub_df = sub_df.loc[sub_df['reward'] == reward]
            sub_df = sub_df.loc[sub_df['GD_batch_size'] == GD_batch_size]
            sub_df = sub_df.loc[sub_df['T2'] == T2]
            sub_df = sub_df.loc[sub_df['dim'] == dim] 
            sub_df = sub_df.loc[sub_df['gamma'] == gamma] 
            sub_df = sub_df.loc[sub_df['alpha'] == alpha] 
            sub_df = sub_df.loc[sub_df['pool_num'] == pool_num]
            sub_df = sub_df.loc[sub_df['batch_size'] == batch_size] 
            sub_df = sub_df.loc[sub_df['num_actions'] == num_actions]

            curve_name = r'$\lambda = $' + '{:.7f}'.format(alg_lambda) + r'$- \beta = $' + '{:.7f}'.format(exp_coef) + r'$- width = $' + '{:.5f}'.format(neuron_per_layer) \
                + r'$- lr = $' + '{:.6f}'.format(lr)  + r'$- T = $' + '{:.5f}'.format(T) + r'$- stop_count = $' + '{:.5f}'.format(stop_count) + \
                    r'$- T2 = $' + '{:.7f}'.format(T2) + r'$- small_loss = $' + '{:.7f}'.format(small_loss) + r'$- pool_size = $' + '{:.7f}'.format(pool_num) + r'$- batch_size = $' + '{:.7f}'.format(batch_size)
            # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
            regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
            top_k =np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['top_k']])
            # picked_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['pick_vars_all']])
            # avg_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['avg_vars']])
            #print('Axes:', axes[row_counter][col_counter])
            #print(np.mean(regrets_bp, axis=0))
            #print(generic_lines[counter])
            print(regrets_bp.shape)
            axes_reg.plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = generic_lines[counter], linestyle = linestyles[counter%13])
            axes_reg.fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.2*np.std(regrets_bp, axis=0),
                                np.mean(regrets_bp, axis=0)+0.2*np.std(regrets_bp, axis=0), alpha=0.2,
                                color = generic_lines[counter])
            axes_topk.plot(top_k_percentages , np.mean(top_k, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = generic_lines[counter], linestyle = linestyles[counter+2%13], marker='o')
            #axes[1,0].fill_between(np.arange(configs['T']), np.mean(top_k, axis=0)-0.2*np.std(top_k, axis=0),
                                #np.mean(top_k, axis=0)+0.2*np.std(top_k, axis=0), alpha=0.2,
                                #color = generic_lines[counter])
            axes_topk.errorbar(top_k_percentages, np.mean(top_k, axis=0), yerr=0.4*np.std(top_k, axis=0), fmt='o', markersize=4, capsize=8, color = generic_lines[counter])

            counter += 1

        #axes[row_counter][col_counter].set_xlabel(r'$t$')
        axes_reg.set_title(f'{net}-UCB')
        axes_topk.set_title(f'{net}-UCB')
        #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
        col_counter += 1

    #lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[0][1],axes[1][0],axes[1][1]]]
    lines_labels = axes_reg.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    fig_reg.legend(lines,labels, loc ='center',bbox_to_anchor=(0.5, -0.4), fancybox = True, ncol = 4)

    lines_labels = axes_topk.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    fig_topk.legend(lines,labels, loc ='center',bbox_to_anchor=(0.5, -0.4), fancybox = True, ncol = 4)
    #fig.tight_layout()

    plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    fig_reg.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_reg.pdf',bbox_inches='tight')
    fig_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_topk.pdf',bbox_inches='tight')

    #plt.show(bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--base_dir', type=str,  default='/cluster/scratch/bsoyuer/base_code/graph_BO/results/')
    parser.add_argument('--exp_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_name', type=str,  default='default_vs_random_rew0')

    args = parser.parse_args()
    main(args)