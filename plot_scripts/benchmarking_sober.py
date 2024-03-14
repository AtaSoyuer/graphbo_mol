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
from torch_geometric.datasets import QM9
import os.path as osp
import os
import sys
sys.path.append(os.path.abspath("../graph_BO")) 
from dataset_class_w_edgeix import MyTransform, Complete 
import torch_geometric.transforms as T
import csv
#DIR = '/local/pkassraie/gnnucb/results/'


'''
CHANGE EXP_NAME PLOT_NAME RESULT_DIR IN ARGS AND ALSO THE REWARD VALUE IN THE CONFIGS DICT BELOW!!!!
.
'''

rews_first = [0,1,2,3,4]
rews_second = [11,12,5,6,7]

sober_rews_first = ['0','1','2','3','4']
sober_rews_second = ['5','6','7','11','12']

bnnbo_rews_first = ['mu','alpha','homo','lumo','gap']
bnnbo_rews_second = ['r2','zpve','u0','cv','u0_atom']

def main(args):
    counter = 0
    row_counter = 0
    rew_counter = 0
    top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_sharpness = [0.5, 0.95]
    alpha_sharpness_str = [str(x) for x in alpha_sharpness]

    bund = bd.icml2022()
    bund['text.usetex'] = False
    bund['font.family'] = 'DejaVu Serif'
    #print('Bund:', bund)

    #plot_name = 'new_paper_hyperparam_{}T_{}actions_QM9'.format(configs['T'], configs['num_actions'])
    plot_name = args.plot_name

    plt.rcParams.update(bundles.icml2022(ncols=2,nrows=5,tight_layout=True))
    #fig_reg, axes_reg = plt.subplots( ncols = 1, nrows=1, (12,9))
    fig_rew_n_topk, axes_rew_n_topk = plt.subplots( ncols = 2, nrows=5,figsize=(10,16))
    #for r in zip(rews_first, sober_rews_first, bnnbo_rews_first): #zip(rews_second, sober_rews_second, bnnbo_rews_second):
    print(os.path.join('/cluster/scratch/bsoyuer/base_code/graph_BO/results/', args.exp_dir))
    #subdirs = [i[0] for i in os.walk(os.path.join('/cluster/scratch/bsoyuer/base_code/graph_BO/results/', args.exp_dir))]
    subdirs = os.listdir(os.path.join('/cluster/scratch/bsoyuer/base_code/graph_BO/results/', args.exp_dir))
    subdirs = sorted(subdirs)
    print(subdirs)
    for exp_folder in subdirs:
        rew_counter = rews_second[row_counter]
        print('rowcounter:', row_counter)
        print('rewcounter:', rew_counter)
        print(os.path.join(args.exp_dir, exp_folder))
        df_full, _ = collect_exp_results(os.path.join(args.exp_dir, exp_folder))
        target = rew_counter
        print(target)

        path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'QM9')
        transform = T.Compose([MyTransform(target), T.AddLaplacianEigenvectorPE(k=1, attr_name=None, is_undirected=True), Complete(), T.Distance(norm=False)])
        dataset = QM9(path, transform=transform)

        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        #dataset.data.y = (dataset.data.y - mean) / std
        mean, std = mean[:, target].item(), std[:, target].item()
    

        indices_to_remove_file = open("/cluster/scratch/bsoyuer/base_code/SOBER/indices_of_smiles_toremove_github.csv", "r")
        indices_to_remove = list(csv.reader(indices_to_remove_file, delimiter=","))[0]
        indices_to_remove_file.close()
        indices_to_remove = [int(item) for item in indices_to_remove]
        #print('indices to remove:', indices_to_remove)

        indices_to_keep = list(set(range(len(dataset))) - set(indices_to_remove))
        MAX_REWARD = np.max(np.array(dataset.data.y)[indices_to_keep,target])
        print('len dataset:',len(indices_to_keep))
        print('max_reward:',MAX_REWARD)

        #print(mean)
        #print(std)

        configurations = [('sober', 200.0), ('sober', 1200.0), ('bnnbo', 5.0), ('bnnbo', 1200.0), ('ucb',True), ('ucb',False)]


        for config in configurations:
            # print(config)
            # print(len(configurations))
            algorithm = config[0]
            if algorithm == 'sober':
                n_init = config[1]
                print('n_init:', n_init)
            if algorithm == 'bnnbo':
                n_start = config[1]
                print('n_start:', n_start)
            if algorithm == 'ucb':
                no_var_computation = config[1]
                print('no_var_comp:', no_var_computation)

            sub_df = df_full.loc[df_full['algorithm'] == algorithm]
            # print(sub_df['rewards'])

            # if (sub_df['algorithm'] == 'sober').all():
            #      sub_df = sub_df.loc[sub_df['n_init'] == n_init]
            if (sub_df['algorithm'] == 'sober').all():
                sub_df_sober = sub_df.loc[sub_df['n_init'] == float(n_init)]
            if (sub_df['algorithm'] == 'bnnbo').all():
                #print('sub_df:',sub_df['n_start'])
                sub_df_bnnbo = sub_df.loc[sub_df['n_start'] == float(n_start)]
                #print('sub_df:',sub_df_bnnbo['n_start'])
            if (sub_df['algorithm'] == 'ucb').all():
                sub_df_ucb = sub_df.loc[sub_df['no_var_computation'] == no_var_computation]
                #print('sub_df_sober:',sub_df_sober['n_init'])

            curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{SOBER} $', r'$\bf{BNN-BO} $', r'$\bf{SOBER-Warmup} $', r'$\bf{BNN-BO-Warmup} $', r'$\bf{SS-RAND} $']

            if (sub_df['algorithm'] == 'ucb').all():
                if (sub_df_ucb['no_var_computation'] == True).all():
                    curve_name = curve_name_prepends[5] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[5]
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[0] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[0]
                    lst = linestyles[0]
            elif (sub_df['algorithm'] == 'sober').all():
                if (sub_df_sober['n_init'] == 1200).all():
                    curve_name = curve_name_prepends[3] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[3]
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[1] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[1]
                    lst = linestyles[0]
            elif (sub_df['algorithm'] == 'bnnbo').all():
                if (sub_df_bnnbo['n_start'] == 1200.0).all():
                    curve_name = curve_name_prepends[4] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[4]
                    lst = linestyles[0]
                else:
                    curve_name = curve_name_prepends[2] #+ curve_name_params
                    print(curve_name)
                    clr = generic_lines[2]
                    lst = linestyles[0]
                    #print('subdf:', sub_df_bnnbo['rewards'])

                    
            # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
            if (sub_df['algorithm'] == 'sober').all():
                top_k = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_sober['top_k']])
                collected_max = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_sober['collected_max']])
            elif (sub_df['algorithm'] == 'bnnbo').all():
                top_k = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_bnnbo['top_k']])
                collected_max = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_bnnbo['collected_max']])
            else:
                top_k = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_ucb['top_k']])
                collected_max = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_ucb['collected_max']])

            # for percentages in sub_df['rewards']:
            #     print(np.array(percentages).shape)
            if (sub_df['algorithm'] == 'sober').all():
                step_sizes = np.array([np.squeeze(np.array(percentages)).shape for percentages in sub_df_sober['rewards']])
                min_step_size = np.min(step_sizes)
                print('min_step_size:', min_step_size)
                rewards_all = np.array([np.squeeze(np.array(percentages)[:min_step_size-1]) for percentages in sub_df_sober['rewards']])
                print('REWS ALL:', rewards_all.shape)
            elif (sub_df['algorithm'] == 'ucb').all():
                rewards_all = np.array([np.squeeze(np.array(percentages))*std+mean for percentages in sub_df_ucb['rewards']])
                print('REWS ALL:', rewards_all.shape)
            elif (sub_df['algorithm'] == 'bnnbo').all():
                rewards_all = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df_bnnbo['rewards']])
                print('REWS ALL:', rewards_all.shape)

            simple_regret = np.zeros((rewards_all.shape[0], rewards_all.shape[1]-1))
            print('SIMPLE_REGRET:', simple_regret.shape)
            print('SHAPE:', np.array([np.max(rewards_all[0,i]).item() for i in range(1,rewards_all.shape[1])]).shape)
            #print('VALUES:', np.array([np.max(rewards_all[0,i]).item() for i in range(1,rewards_all.shape[1])])[:20])

            for row in range(rewards_all.shape[0]):
                simple_regret[row,:] = np.array([np.max(rewards_all[row,:i]) for i in range(1,rewards_all.shape[1])])


            axes_rew_n_topk[row_counter][0].plot(np.mean(simple_regret, axis=0), linestyle['GNN'+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
            axes_rew_n_topk[row_counter][0].fill_between(np.arange(simple_regret.shape[1]), np.mean(simple_regret, axis=0)-0.8*np.std(simple_regret, axis=0),
                                np.mean(simple_regret, axis=0)+0.8*np.std(simple_regret, axis=0), alpha=0.2,
                                color = clr)
            axes_rew_n_topk[row_counter][0].axhline(y = MAX_REWARD, color = 'dimgray', linestyle = '-', linewidth=2)
            axes_rew_n_topk[row_counter][0].grid(alpha=0.8)
            
            axes_rew_n_topk[row_counter][1].plot(top_k_percentages , np.mean(top_k, axis=0), linestyle['GNN'+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=4)
            #axes[1,0].fill_between(np.arange(configs['T']), np.mean(top_k, axis=0)-0.2*np.std(top_k, axis=0),
                                #np.mean(top_k, axis=0)+0.2*np.std(top_k, axis=0), alpha=0.2,
                                #color = generic_lines[counter])
            axes_rew_n_topk[row_counter][1].errorbar(top_k_percentages, np.mean(top_k, axis=0), yerr=0.4*np.std(top_k, axis=0), fmt='o', markersize=5, capsize=6, color = clr)
            axes_rew_n_topk[row_counter][1].set_xscale('log')
            axes_rew_n_topk[row_counter][1].grid(alpha=0.8)


            print(f'Collected global max {np.sum(collected_max)} out of {len(collected_max)} times, ratio={np.sum(collected_max)/len(collected_max)}')
            print(f'TopK Means: {np.mean(top_k, axis=0)}')
            print(f'TopK Stds: {np.std(top_k, axis=0)}')
                
                
        #axes[row_counter][col_counter].set_xlabel(r'$t$')

        rewards_names = [r'Dipole moment$(\mu)$',r'Isotropic polarizability$(\alpha)$',r'Highest occupied molecular orbital energy$(\epsilon_{HOMO})$',
            r'Lowest unoccupied molecular orbital energy$(\epsilon_{LUMO})$',r'Gap Between $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$$(\Delta \epsilon)$',
            r'Electronic spatial extent$(\langle R^2 \rangle)$',r'Zero point vibrational energy(ZPVE)',r'Internal energy at 0K$(U_0)$',r'Internal energy at 298.15K$(U)$',
            r'Enthalpy at 298.15K$(H)$',r'Free energy at 298.15K$(G)$',r'Heat capacity at 298.15K$(c_{v})$',r'Atomization energy at 0K$(U_0^{ATOM})$',
            r'Atomization energy at 298.15K$(U^{ATOM})$',r'Atomization enthalpy at 298.15K$(H^{ATOM})$',r'Atomization free energy at 298.15K$(G^{ATOM})$',
            r'Rotational constant A',r'Rotational constant B',r'Rotational constant C']
        #print(reward)
        #fig_rew_n_topk.suptitle(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
        axes_rew_n_topk[row_counter][0].set_title(r'$QM9:$'+rewards_names[rew_counter],)#fontsize=20)
        axes_rew_n_topk[row_counter][1].set_title(r'$QM9:$'+rewards_names[rew_counter],)#fontsize=20)
        #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
        #axes_rew_n_topk[row_counter][counter%2].set(ylabel=r'Calibration of Confidence Intervals', xlabel=r'$\alpha$')
        axes_rew_n_topk[row_counter][0].set(xlabel=r'Online Oracle Calls', ylabel=r'$top\_1$')
        axes_rew_n_topk[row_counter][1].set(xlabel=r'Log-TopK Percent Samples Evaluated', ylabel=r"Ratio of Top-K Samples Acquired")
        #print(reward)
        lines_labels = [ax.get_legend_handles_labels() for ax in [axes_rew_n_topk[0][0], axes_rew_n_topk[1][1]]] #axes_rew_n_topk[2]]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #print(lines)
        #print(labels)
        fig_rew_n_topk.legend(lines[:6],labels[:6], loc ='lower center',bbox_to_anchor=(0.55, -0.05), fancybox = True, ncol = 8, prop = { "size": 14 } )

        row_counter += 1

    #fig.tight_layout()
    #plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    print('here3')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{args.plot_name}_rew.pdf',bbox_inches='tight')
    print('here4')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{args.plot_name}_topk.pdf',bbox_inches='tight')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{args.plot_name}_topk.svg',bbox_inches='tight',format='svg')
    fig_rew_n_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{args.plot_name}_rew.svg',bbox_inches='tight',format='svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--base_dir', type=str,  default='/cluster/scratch/bsoyuer/base_code/graph_BO/results/')
    parser.add_argument('--exp_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_name', type=str,  default='default_vs_random_rew0')

    args = parser.parse_args()
    main(args)