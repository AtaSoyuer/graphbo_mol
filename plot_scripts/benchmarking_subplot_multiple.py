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
#DIR = '/local/pkassraie/gnnucb/results/'


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
    df = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
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
    fig_rew_n_topk, axes_rew_n_topk = plt.subplots( ncols = 2, nrows=5,figsize=(10,16))


    row_counter = 0
    col_counter =0
    for net in ['GNN']:
        counter = 0
        df_net = df.loc[df['net'] == net]
        #print(df_net)
        df_full, _ = collect_exp_results(exp_name=args.exp_dir)
        df_new = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
        for r in [0,1,2,3,4]:#[0,1,2,3,4]:#[5,6,7,11,12]: #[0,1,2,3,4]
            df_net = df_new.loc[df_new['reward'] == r]
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
                            df_net['alternative'],
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
                            df_net['no_var_computation'],
                            df_net['oracle'],
                            df_net['rand'],
                            df_net['thompson_sampling'],
                            #df_net['pretrain_model_name'],
                            ) if
                            not all(z == config[0] for z in config[1:])]
            configurations = list(set(configurations))


            #print(len(configurations))
            # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
            for config in configurations:
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
                reward = config[0]
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
                alternative = config[1]
                no_var_computation = config[2]
                oracle = config[3]
                rand = config[4]
                thompson_sampling = config[5]
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
                sub_df = df_net.loc[df_net['no_var_computation'] == no_var_computation]
                sub_df = sub_df.loc[sub_df['oracle'] == oracle]
                sub_df = sub_df.loc[sub_df['rand'] == rand]
                sub_df = sub_df.loc[sub_df['thompson_sampling'] == thompson_sampling]
                #sub_df = sub_df.loc[sub_df['pretrain_model_name'] == pretrain_model_name]

                curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{GNN-SS-GREEDY} $', r'$\bf{GNN-SS-TS} $', r'$\bf{SS-RANDOM} $', r'$\bf{SS-ORACLE} $']
            
                #print(sub_df['pretrain_model_name'])
                #if (sub_df['pretrain_model_name'] == 'nnconv_reward3n4_8000samples_100ep').all():
                if (sub_df['no_var_computation'] == True).all() and (sub_df['oracle'] == False).all() and (sub_df['rand'] == False).all():
                    curve_name = curve_name_prepends[1] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[1]
                    lst = linestyles[0]
                elif (sub_df['oracle'] == True).all():
                    curve_name = curve_name_prepends[4] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[4]
                    lst = linestyles[0]
                elif (sub_df['rand'] == True).all():
                    curve_name = curve_name_prepends[3] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[3]
                    lst = linestyles[0]
                elif (sub_df['thompson_sampling'] == True).all():
                    curve_name = curve_name_prepends[2] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[2]
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[0] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[0]
                    lst = linestyles[0]

                dataset = QM9(root="../../data/QM9")

                TARGET = np.unique(reward)
                print('TARGET:',TARGET)

                mean = dataset.data.y.mean(dim=0, keepdim=True)
                std = dataset.data.y.std(dim=0, keepdim=True)
                mean, std = mean[:, TARGET].item(), std[:, TARGET].item()

                row_list = []
                import csv
                with open('/cluster/scratch/bsoyuer/base_code/SOBER/removed.txt', 'r') as fd:
                    rows = (line.split() for line in fd)
                    count = 0
                    for row in rows:
                        if count > 8 and count <= 3062:
                            row_list.append(row[0])
                        count += 1
                
                indices_to_keep = list(set(range(len(dataset))) - set(row_list))
                MAX_REWARD = np.max(np.array(dataset.data.y)[indices_to_keep,TARGET])
                        
                # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
                regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
                top_k = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['top_k']])
                coverages = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['coverages']])
                collected_max = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['collected_max']])

                rewards_all = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['rewards']])*std+mean
                print('REWS ALL:', regrets_bp.shape)

                simple_regret = np.zeros((rewards_all.shape[0], rewards_all.shape[1]-1))

                for row in range(rewards_all.shape[0]):
                    simple_regret[row,:] = np.array([np.max(rewards_all[row,:i]) for i in range(1,rewards_all.shape[1])])

                
                if curve_name == curve_name_prepends[4]:

                    axes_rew_n_topk[row_counter][0].plot(np.mean(simple_regret, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    axes_rew_n_topk[row_counter][0].fill_between(np.arange(simple_regret.shape[1]), np.mean(simple_regret, axis=0), MAX_REWARD, alpha=0.2,
                                        color = clr)
                    axes_rew_n_topk[row_counter][0].axhline(y = MAX_REWARD, color = 'dimgray', linestyle = '-', linewidth=2)
                    # axes_rew_n_topk[0].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    # axes_rew_n_topk[0].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.2*np.std(regrets_bp, axis=0),
                    #                     np.mean(regrets_bp, axis=0)+0.2*np.std(regrets_bp, axis=0), alpha=0.2,
                    #                     color = clr)
                    axes_rew_n_topk[row_counter][0].grid(alpha=0.8)

                    # axes_rew_n_topk[2].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    # axes_rew_n_topk[2].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0), 0, alpha=0.2,
                    #                 color = clr)
                    # axes_rew_n_topk[2].grid(alpha=0.8)

                
                else:

                    axes_rew_n_topk[row_counter][0].plot(np.mean(simple_regret, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    axes_rew_n_topk[row_counter][0].fill_between(np.arange(simple_regret.shape[1]), np.mean(simple_regret, axis=0)-0.8*np.std(simple_regret, axis=0),
                                        np.mean(simple_regret, axis=0)+0.8*np.std(simple_regret, axis=0), alpha=0.2,
                                        color = clr)
                    axes_rew_n_topk[row_counter][0].axhline(y = MAX_REWARD, color = 'dimgray', linestyle = '-', linewidth=2)
                    axes_rew_n_topk[row_counter][0].grid(alpha=0.8)
                    # axes_rew_n_topk[2].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    # axes_rew_n_topk[2].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.4*np.std(regrets_bp, axis=0),
                    #                     np.mean(regrets_bp, axis=0)+0.4*np.std(regrets_bp, axis=0), alpha=0.2,
                    #                     color = clr)
                    # axes_rew_n_topk[2].grid(alpha=0.8)

                    # axes_rew_n_topk[1].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                    # axes_rew_n_topk[1].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.4*np.std(regrets_bp, axis=0),
                    #                 np.mean(regrets_bp, axis=0)+0.4*np.std(regrets_bp, axis=0), alpha=0.2,
                    #                 color = clr)
                    # axes_rew_n_topk[1].grid(alpha=0.8)
                
                axes_rew_n_topk[row_counter][1].plot(top_k_percentages , np.mean(top_k, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=4)
                axes_rew_n_topk[row_counter][1].errorbar(top_k_percentages, np.mean(top_k, axis=0), yerr=0.6*np.std(top_k, axis=0), fmt='o', markersize=5, capsize=6, color = clr)
                axes_rew_n_topk[row_counter][1].set_xscale('log')
                #axes_rew_n_topk[1].set_ylim(-0.01,1.0)
                axes_rew_n_topk[row_counter][1].grid(alpha=0.8)

                # if curve_name != curve_name_prepends[1]:
                #     axes_rew_n_topk[row_counter][counter%2].plot(alphas, np.mean(coverages, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
                #     axes_rew_n_topk[row_counter][counter%2].fill_between(alphas, np.mean(coverages, axis=0)-np.std(coverages, axis=0),
                #                         np.mean(coverages, axis=0)+np.std(coverages, axis=0), alpha=0.2,
                #                         color = clr)
                #     axes_rew_n_topk[row_counter][counter%2].plot([0.0,1.0], [0.0,1.0], color='black', linewidth=0.5, alpha=0.1)
                #     axes_rew_n_topk[row_counter][counter%2].grid(alpha=0.8)


                print(f'Collected global max {np.sum(collected_max)} out of {len(collected_max)} times, ratio={np.sum(collected_max)/len(collected_max)}')
                print(f'TopK Means: {np.mean(top_k, axis=0)}')
                print(f'TopK Stds: {np.std(top_k, axis=0)}')
                

            rewards_names = [r'Dipole moment$(\mu)$',r'Isotropic polarizability$(\alpha)$',r'Highest occupied molecular orbital energy$(\epsilon_{HOMO})$',
            r'Lowest unoccupied molecular orbital energy$(\epsilon_{LUMO})$',r'Gap Between $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$$(\Delta \epsilon)$',
            r'Electronic spatial extent$(\langle R^2 \rangle)$',r'Zero point vibrational energy(ZPVE)',r'Internal energy at 0K$(U_0)$',r'Internal energy at 298.15K$(U)$',
            r'Enthalpy at 298.15K$(H)$',r'Free energy at 298.15K$(G)$',r'Heat capacity at 298.15K$(c_{v})$',r'Atomization energy at 0K$(U_0^{ATOM})$',
            r'Atomization energy at 298.15K$(U^{ATOM})$',r'Atomization enthalpy at 298.15K$(H^{ATOM})$',r'Atomization free energy at 298.15K$(G^{ATOM})$',
            r'Rotational constant A',r'Rotational constant B',r'Rotational constant C']
            #print(reward)
            #fig_rew_n_topk.suptitle(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
            axes_rew_n_topk[row_counter][0].set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
            axes_rew_n_topk[row_counter][1].set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
            #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
            #axes_rew_n_topk[row_counter][counter%2].set(ylabel=r'Calibration of Confidence Intervals', xlabel=r'$\alpha$')
            axes_rew_n_topk[row_counter][0].set(xlabel=r'Online Oracle Calls', ylabel=r'$top\_1$')
            axes_rew_n_topk[row_counter][1].set(xlabel=r'Log-TopK Percent Samples Evaluated', ylabel=r"Ratio of Top-K Samples Acquired")

            #axes_rew_n_topk[2].set(xlabel=r'Online Oracle Calls', ylabel=r'$R_t$')
            #axes_rew_n_topk[0].set_title(r'$QM9:$'+rewards_names[reward],)
            print('row, counter: ', row_counter, counter)
            #col_counter += 1
            #if counter%2 == 1:
            row_counter += 1
            #counter += 1

    lines_labels = [ax.get_legend_handles_labels() for ax in [axes_rew_n_topk[0][0], axes_rew_n_topk[1][1]]] #axes_rew_n_topk[2]]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #print(lines)
    #print(labels)
    fig_rew_n_topk.legend(lines[:5],labels[:5], loc ='lower center',bbox_to_anchor=(0.55, -0.05), fancybox = True, ncol = 8, prop = { "size": 14 } )
    #fig_sha.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1), prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    print('here2')
    #fig.tight_layout()
    #plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    print('here3')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew_n_topk.pdf',bbox_inches='tight')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew_n_topk.svg',bbox_inches='tight',format='svg')
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