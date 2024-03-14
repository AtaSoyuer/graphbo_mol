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
from matplotlib.legend_handler import HandlerTuple
from tueplots import bundles as bd
import torch_geometric
from torch_geometric.datasets import QM9
import os
import json
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import pylab


'''
CHANGE EXP_NAME IN THE ABOVE LINE TO READ FROM DIFFERENT DIRECTORY

IF YOU WANT TO SELECT JSON FILES WITH ALL CONTAINING SPECIFIC VALUES FOR CERTAIN HYPERPARAMS,
YOU CAN ADD THOSE VALUES TO CONFIGS BELOW AND SELECT THOSE PARTICULAR VALUES BEFOREHAND AS WE DO BY DEFAULT FOR
FEAT DIM IN LINES ~84

IN THE MAIN LOOP, WE DIVIDE ALL SELECTED EXPERIMENTS WITH RESPECT TO UNIQUE HYPERPARAM CONFIGURATIONS, SO
IF YOU WANT EXPERIMENTS TO BE GROUPED BY VARIATIONS IN VALUES OF CERTAIN HYPERPARAMS, ONLY USE THOSE!!
.
'''

configs = {
    # Dataset
    'num_nodes': 5, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 12, # 10 or 100 #CHANGE WHEN SYNTHETIC!!!!
    'num_actions': 130831, # any number below 10000 works.
    #BOOLEAN ARGS:
    'alternative':'false',
    'batch_GD':'true',
    'pool':'true',
    'load_pretrained':'true',
    'large_scale':'false',
    'ucb_wo_replacement':'true',
    'focal_loss':'true',
    'pool_top_means':'false',
    'batch_window':'false',
    'small_net_var': 'true',
    'initgrads_on_fly': 'false',
    'no_var_computation': 'true',
    'oracle': 'false',
    'select_K_together': 'false',
    'laplacian_features': 'false',
    'pretraining_load_pretrained': 'false',
    'remove_smiles_for_sober': 'true',
    'runner_verbose': 'false',
    'thompson_sampling': 'false',
    'rand': 'true',
    # GNN-UCB
    'GD_batch_size':50,
    'T':1501,
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
    'pretrain_steps': 100,
    'pretrain_num_indices': 1400,
    'pretrain_model_name': 'nnconv_reward0_4000samples_100ep',
    'pretrain_indices_name': 'pretrain_indices_rew0',
    'pretraining_pretrain_model_name': 'nnconv_reward3n4_8000samples_100ep',
    'pool_num':300,
    'print_every':400,
    'patience':5,
    'select_K': 10,
    'stop_count' : 9000,
    'small_loss' : 1e-4,
    'subsample_method':'random', 
    'subsample_num':20, 
    'synthetic':0,
    'relative_improvement' : 1e-8,
    'reward':0,
    'reward_plot_dir':0,
}


def main(args):
    top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_sharpness = [0.5, 0.95]
    keys = ['SS_GNN_UCB', 'SS_GNN_GREEDY', 'SS_GNN_ORACLE', 'SS_GNN_RANDOM']
    dicts = {'SS_GNN_UCB':None, 'SS_GNN_GREEDY':None, 'SS_GNN_ORACLE':None, 'SS_GNN_RANDOM':None}
    alpha_sharpness_str = [str(x) for x in alpha_sharpness]
    DIR = args.base_dir
    df_full, _ = collect_exp_results(exp_name=args.exp_dir)

    #Pick which QM9
    #df = df_full.loc[df_full['num_actions']==configs['num_actions']]
    #df = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
    #df = df.loc[df['T'] == configs['T']]
    #print(df)

    bund = bd.icml2022()
    bund['text.usetex'] = False
    bund['font.family'] = 'DejaVu Serif'
    #print('Bund:', bund)



    #plot_name = 'new_paper_hyperparam_{}T_{}actions_QM9'.format(configs['T'], configs['num_actions'])
    plot_name = args.plot_name

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=2,nrows=5,tight_layout=True))
    #fig_reg, axes_reg = plt.subplots( ncols = 1, nrows=1, (12,9))
    fig_cal, axes_cal = plt.subplots( ncols = 2, nrows=5,figsize=(10,16))


    row_counter = 0
    col_counter =0
    counter = 0
    #print(df_net)
    df_full, _ = collect_exp_results(exp_name=args.exp_dir)
    #df_new = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
    for r in ['0','1','2','3','4','5','6','7','11','12']: #[0,1,2,3,4]
        target = r
        df_net = df_full.loc[df_full['reward'] == r]
        configurations = [config for config in
                        zip(
                        #df_net['alg_lambda'], 
                        #df_net['exploration_coef'], 
                        #df_net['pretrain_steps'], 
                        #df_net['neuron_per_layer'], 
                        #df_net['lr'], 
                        #df_net['stop_count'], 
                        #df_net['T'], 
                        #df_net['small_loss'], 
                        df_net['reward'],
                        #df_net['GD_batch_size'], 
                        #df_net['patience'],
                        #df_net['T2'], 
                        #df_net['dim'], 
                        #df_net['gamma'], 
                        #df_net['alpha'],
                        #df_net['pool_num'],
                        #df_net['patience'], 
                        #df_net['batch_size'],
                        #df_net['patience'], 
                        #df_net['num_actions'], 
                        #df_net['alternative'],
                        #df_net['batch_GD'], 
                        #df_net['pool'], 
                        #df_net['load_pretrained'], 
                        #df_net['large_scale'], 
                        #df_net['ucb_wo_replacement'], 
                        #df_net['focal_loss'], 
                        #df_net['patience'],
                        #df_net['pool_top_means'],
                        #df_net['batch_window_size'],
                        #df_net['batch_window'],
                        #df_net['no_var_computation'],
                        #df_net['oracle'],
                        #df_net['rand'],
                        #df_net['thompson_sampling'],
                        df_net['algorithm'],
                        #df_net['pretrain_model_name'],
                        ) if
                        not all(z == config[0] for z in config[1:])]
        configurations = list(set(configurations))
        print('configurations:', configurations)


        #print(len(configurations))
        # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
        for config in configurations:
            net = 'GNN'
            print('LEN CONFGIS:', len(configurations))
            # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
            # alg_lambda = config[0]
            # exp_coef = config[1]
            # pretrain_steps = config[2]
            # neuron_per_layer = config[3]
            # lr = config[4]
            # stop_count = config[5]
            #T = config[0]
            # small_loss = config[7]
            reward = int(config[0])
            # GD_batch_size = config[9]
            #T2 = config[2]
            # dim = config[11]
            # gamma = config[12]
            # alpha = config[13]
            # pool_num = config[14]
            # batch_size = config[15]
            # num_actions = config[16]
            # alternative = config[17]
            # batch_GD = config[18]
            # pool = config[19]
            # load_pretrained = config[20]
            # large_scale = config[21]
            # ucb_wo_replacement = config[22]
            # focal_loss = config[23]
            # pool_top_means = config[24]
            # batch_window_size = config[25]
            # batch_window = config[26]
            #alternative = config[1]
            #no_var_computation = config[2]
            #oracle = config[3]
            #rand = config[4]
            #thompson_sampling = config[2]
            algorithm = config[1]
            #pretrain_model_name = config[4]

            #sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
            # sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]
            # sub_df = sub_df.loc[sub_df['neuron_per_layer'] == neuron_per_layer]
            # sub_df = sub_df.loc[sub_df['lr'] == lr] 
            # sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
            # sub_df = sub_df.loc[sub_df['stop_count'] == stop_count] 
            # sub_df = sub_df.loc[sub_df['T'] == T] 
            # sub_df = sub_df.loc[sub_df['small_loss'] == small_loss]
            #sub_df = df_net.loc[df_net['reward'] == reward]
            # sub_df = sub_df.loc[sub_df['GD_batch_size'] == GD_batch_size]
            #sub_df = df_net.loc[df_net['T2'] == T2]
            # sub_df = sub_df.loc[sub_df['dim'] == dim] 
            # sub_df = sub_df.loc[sub_df['gamma'] == gamma] 
            # sub_df = sub_df.loc[sub_df['alpha'] == alpha] 
            # sub_df = sub_df.loc[sub_df['pool_num'] == pool_num]
            # sub_df = sub_df.loc[sub_df['batch_size'] == batch_size] 
            # sub_df = sub_df.loc[sub_df['num_actions'] == num_actions]
            # sub_df = sub_df.loc[sub_df['alternative'] == alternative]
            # sub_df = sub_df.loc[sub_df['batch_GD'] == batch_GD]
            # sub_df = sub_df.loc[sub_df['pool'] == pool]
            # sub_df = sub_df.loc[sub_df['load_pretrained'] == load_pretrained]
            # sub_df = sub_df.loc[sub_df['large_scale'] == large_scale]
            # sub_df = sub_df.loc[sub_df['ucb_wo_replacement'] == ucb_wo_replacement]
            # sub_df = sub_df.loc[sub_df['focal_loss'] == focal_loss]
            # sub_df = sub_df.loc[sub_df['pool_top_means'] == pool_top_means]
            # sub_df = sub_df.loc[sub_df['batch_window_size'] == batch_window_size]
            # sub_df = sub_df.loc[sub_df['batch_window'] == batch_window]
            sub_df = df_net.loc[df_net['algorithm'] == algorithm]
            #sub_df = sub_df.loc[sub_df['collected_max'] == True]
            #sub_df = sub_df.loc[sub_df['pretrain_model_name'] == pretrain_model_name]

            curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{GNN-SS-TS} $', r'$\bf{GP-SS-UCB} $', r'$\bf{GP-SS-TS} $', r'$\bf{SOBER} $']
        
            #print(sub_df['pretrain_model_name'])
            #if (sub_df['pretrain_model_name'] == 'nnconv_reward3n4_8000samples_100ep').all():
            if (sub_df['algorithm'] == 'ucb').all():
                if (sub_df['thompson_sampling'] == True).all():
                    curve_name = curve_name_prepends[1] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[1]
                    print(clr)
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[0] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[0]
                    print(clr)
                    lst = linestyles[0]
            elif (sub_df['algorithm'] == 'gp').all():
                if (sub_df['thompson_sampling'] == True).all():
                    curve_name = curve_name_prepends[3] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[3]
                    print(clr)
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[2] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[2]
                    print(clr)
                    lst = linestyles[0]
            elif (sub_df['algorithm'] == 'sober').all():
                curve_name = curve_name_prepends[4] #+ curve_name_params
                print(curve_name)
                clr = generic_lines[4]
                print(clr)
                lst = linestyles[0]

            dataset = QM9(root="../../data/QM9")

            TARGET = np.unique(reward)
            print('TARGET:',TARGET)

            mean = dataset.data.y.mean(dim=0, keepdim=True)
            std = dataset.data.y.std(dim=0, keepdim=True)
            mean, std = mean[:, TARGET].item(), std[:, TARGET].item()

            graph_rewards = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['graph_rewards']])[0]
            means = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['means']])[0]
            indices = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['indices']])
            print('means shape:', means.shape)
            print('rews shape:', graph_rewards.shape)


            max_val = torch.max(torch.tensor([torch.max(torch.tensor(graph_rewards)), torch.max(torch.tensor(means))]))
            min_val = torch.min(torch.tensor([torch.min(torch.tensor(graph_rewards)), torch.min(torch.tensor(means))]))
            #print(torch.tensor(means).shape)
            #print(torch.tensor(graph_rewards).shape)
            #print(max_val)
            #conf_bounds = np.sqrt(args.exploration_coef)*np.array([learner.get_post_var(idx) for idx in range(args.num_actions)])
            cmap = matplotlib.colors.ListedColormap(['red', 'green'])
            #plt.errorbar(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), xerr=conf_bounds, fmt='o', alpha=0.2)
            axes_cal[row_counter][col_counter%2].scatter(torch.tensor(means), torch.tensor(graph_rewards), \
                        c=np.isin(np.arange(130480), indices), cmap=cmap, s=1/2)
            axes_cal[row_counter][col_counter%2].plot([min_val, max_val], [min_val, max_val], alpha=0.3)

            rewards_names = [r'Dipole moment$(\mu)$',r'Isotropic polarizability$(\alpha)$',r'Highest occupied molecular orbital energy$(\epsilon_{HOMO})$',
            r'Lowest unoccupied molecular orbital energy$(\epsilon_{LUMO})$',r'Gap Between $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$$(\Delta \epsilon)$',
            r'Electronic spatial extent$(\langle R^2 \rangle)$',r'Zero point vibrational energy(ZPVE)',r'Internal energy at 0K$(U_0)$',r'Internal energy at 298.15K$(U)$',
            r'Enthalpy at 298.15K$(H)$',r'Free energy at 298.15K$(G)$',r'Heat capacity at 298.15K$(c_{v})$',r'Atomization energy at 0K$(U_0^{ATOM})$',
            r'Atomization energy at 298.15K$(U^{ATOM})$',r'Atomization enthalpy at 298.15K$(H^{ATOM})$',r'Atomization free energy at 298.15K$(G^{ATOM})$',
            r'Rotational constant A',r'Rotational constant B',r'Rotational constant C']
            #print(reward)
            #fig_rew_n_topk.suptitle(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
            axes_cal[row_counter][col_counter%2].set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
            axes_cal[row_counter][col_counter%2].set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)

            #axes_rew_n_topk[2].set(xlabel=r'Online Oracle Calls', ylabel=r'$R_t$')
            #axes_rew_n_topk[0].set_title(r'$QM9:$'+rewards_names[reward],)
            print('row, counter: ', row_counter, counter)
            if col_counter%2 == 1:
                row_counter += 1

            col_counter += 1
        #counter += 1

    # lines_labels = [ax.get_legend_handles_labels() for ax in [axes_cal[0][0], axes_cal[1][1]]] #axes_rew_n_topk[2]]]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # #print(lines)
    # #print(labels)
    # fig_cal.legend(lines[:4],labels[:4], loc ='lower center',bbox_to_anchor=(0.55, -0.05), fancybox = True, ncol = 8, prop = { "size": 14 } )
    # #fig_sha.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1), prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    print('here2')
    #fig.tight_layout()
    #plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    print('here3')
    fig_cal.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew_n_topk.pdf',bbox_inches='tight')
    fig_cal.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew_n_topk.svg',bbox_inches='tight',format='svg')
    #plt.show(bbox_inches='tight')
    
    with open('plot_data_mu.json', 'w') as fp:
        json.dump(dicts, fp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--base_dir', type=str,  default='/cluster/scratch/bsoyuer/base_code/graph_BO/results/')
    parser.add_argument('--exp_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_name', type=str,  default='default_vs_random_rew0')

    args = parser.parse_args()
    main(args)