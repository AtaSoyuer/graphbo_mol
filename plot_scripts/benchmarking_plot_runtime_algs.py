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
#DIR = '/local/pkassraie/gnnucb/results/'


'''
CHANGE EXP_NAME PLOT_NAME RESULT_DIR IN ARGS AND ALSO THE REWARD VALUE IN THE CONFIGS DICT BELOW!!!!
.
'''

configs = {
    'T':1501,
    'reward': 0,
}

target = configs['reward']
print(target)

path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'QM9')
transform = T.Compose([MyTransform(target), T.AddLaplacianEigenvectorPE(k=1, attr_name=None, is_undirected=True), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform)

mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

print(mean)
print(std)


def main(args):
    top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_sharpness = [0.5, 0.95]
    alpha_sharpness_str = [str(x) for x in alpha_sharpness]
    DIR = args.base_dir
    df, _ = collect_exp_results(exp_name=args.exp_dir)
    #print(df.loc[:,'time'])
    # print(df)

    bund = bd.icml2022()
    bund['text.usetex'] = False
    bund['font.family'] = 'DejaVu Serif'
    bund['lines.linewidth'] = 1.0
    #print('Bund:', bund)


    #plot_name = 'new_paper_hyperparam_{}T_{}actions_QM9'.format(configs['T'], configs['num_actions'])
    plot_name = args.plot_name


    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_topk, axes_topk = plt.subplots( ncols = 1, nrows=1, )
    fig_run, axes_run = plt.figure(), plt.axes()

    row_counter = 0
    col_counter =0

    counter = 0

    configurations = [config for config in df['algorithm'] if not all(z == config[0] for z in config[1:])]
    configurations = list(set(configurations))
    print(configurations)
    # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():

    # df_sober = df.loc[df['algorithm'] == 'sober']
    # configurations_sober = [config for config in zip(df_sober['algorithm'], df_sober['n_init']) if not all(z == config[0] for z in config[1:])]
    # configurations = [('ucb', 0.0)]
    # configurations_sober = list(set(configurations_sober))
    # configurations.extend(configurations_sober)
    # print('CONFIGS:', configurations)

    for config in configurations:
        print(config)
        #print(len(configurations))
        algorithm = config
        #n_init = config[1]
        #print('n_init:', n_init)

        sub_df = df.loc[df['algorithm'] == algorithm]

        # if (sub_df['algorithm'] == 'sober').all():
        #      sub_df = sub_df.loc[sub_df['n_init'] == n_init]


        curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{GNN-SS-GREEDY} $', r'$\bf{SOBER} $', r'$\bf{DeepBNN-BO} $',]

        if (sub_df['algorithm'] == 'ucb').all():
            for b in [True,False]:
                sub_df_ucb = sub_df.loc[df['alternative'] == b]
                #print(sub_df_ucb['T0'].iloc[0])
                if (sub_df_ucb['algorithm'] == 'ucb').all():
                    if (sub_df_ucb['alternative'] == True).all():
                        curve_name = curve_name_prepends[0]+r' $batch\_size=$'+str(sub_df_ucb['batch_size'].iloc[0])#+ curve_name_params
                        print(curve_name)
                        clr = generic_lines[0]
                        lst = linestyles[0]
                    else:
                        #print('TRUE NO VARS')
                        curve_name = curve_name_prepends[1]+r' $batch\_size=$'+str(sub_df_ucb['batch_size'].iloc[0]) #+ curve_name_params
                        print(curve_name)
                        clr = generic_lines[1]
                        lst = linestyles[1]

                if (sub_df['algorithm'] == 'ucb').all():
                    time_all = np.array([np.squeeze(np.array(k)) for k in sub_df_ucb['time']])
                    print('TIME ALL:', time_all.shape)

                times = time_all
                
                if (sub_df['algorithm'] == 'ucb').all(): 
                    axes_run.plot(np.arange(times[:,int(sub_df_ucb['T0'].iloc[0]):].shape[1])+1 , np.mean(times[:,int(sub_df_ucb['T0'].iloc[0]):], axis=0), linestyle['GNN'+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=2, linewidth=1.0)
                    axes_run.grid(alpha=0.8)
                    axes_run.set_yscale('log')

                counter += 1
        else:

            if (sub_df['algorithm'] == 'sober').all():
                curve_name = curve_name_prepends[2]+r' $batch\_size=$'+str(sub_df['batch_size'].iloc[0]) #+ curve_name_params
                print(curve_name)
                clr = generic_lines[2]
                lst = linestyles[2]
            elif (sub_df['algorithm'] == 'bnnbo').all():
                curve_name = curve_name_prepends[3]+r' $batch\_size=200$'#+ curve_name_params
                print(curve_name)
                clr = generic_lines[3]
                lst = linestyles[3]
            if (sub_df['algorithm'] == 'sober').all():
                step_sizes = np.array([np.squeeze(np.array(k)).shape for k in sub_df['time']])
                min_step_size = np.min(step_sizes)
                print('min_step_size:', min_step_size)
                time_all = np.array([np.squeeze(np.array(k)[:min_step_size-1]) for k in sub_df['time']])
                print('TIME ALL:', time_all.shape)
            if (sub_df['algorithm'] == 'bnnbo').all():
                time_all_per_step = np.array([np.squeeze(np.array(k)) for k in sub_df['time']])[0]
                print(time_all_per_step)
                print("bnnbo times shape:", time_all_per_step.shape)
                time_all = np.array([np.sum(k) for k in np.array_split(time_all_per_step, time_all_per_step.shape[0]//200+1)])
                print('TIME ALL:', time_all.shape)

            times = time_all
            
            if (sub_df['algorithm'] == 'sober').all(): 
                axes_run.plot(np.arange(9)+1 , np.mean(times, axis=0)[:9], linestyle['GNN'+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=2, linewidth=1.0)
                axes_run.grid(alpha=0.8)
                axes_run.set_yscale('log')
            elif (sub_df['algorithm'] == 'bnnbo').all(): 
                print(times.shape[0])
                print(times)
                axes_run.plot(np.arange(times.shape[0])+1 , times, linestyle['GNN'+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=2, linewidth=1.0)
                axes_run.grid(alpha=0.8)
                axes_run.set_yscale('log')

            counter += 1
        
            
    #axes[row_counter][col_counter].set_xlabel(r'$t$')

    # rewards_names = [r'$\bf{Dipole\ moment(\mu)}$',r'$\bf{Isotropic\ polarizability(\alpha)}$',r'$\bf{Highest\ occupied\ molecular\ orbital\ energy(\epsilon_{HOMO})}$',
    # r'$\bf{Lowest\ unoccupied\ molecular\ orbital\ energy(\epsilon_{LUMO})}$',r'$\bf{Gap\ Between\ \epsilon_{HOMO}\ and\ \epsilon_{LUMO}(\Delta \epsilon)}$',
    # r'$\bf{Electronic\ spatial\ extent(\langle R^2 \rangle)}$',r'$\bf{Zero\ point\ vibrational\ energy(ZPVE)}$',r'$\bf{Internal\ energy\ at\ 0K(U_0)}$',r'$\bfInternal\ energy\ at\ 298.15K(U)}$',
    # r'$\bf{Enthalpy\ at\ 298.15K(H)}$',r'$\bf{Free\ energy\ at\ 298.15K(G)}$',r'$\bf{Heat\ capacity\ at\ 298.15K(c_{v})}$',r'$\bf{Atomization\ energy\ at\ 0K(U_0^{ATOM})}$',
    # r'$\bf{Atomization\ energy\ at\ 298.15K(U^{ATOM})}$',r'$\bf{Atomization\ enthalpy\ at\ 298.15K(H^{ATOM})}$',r'$\bf{Atomization\ free\ energy\ at\ 298.15K(G^{ATOM})}$',
    # r'$\bf{Rotational\ constant\ A}$',r'$\bf{Rotational\ constant\ B}$',r'$\bf{Rotational\ constant\ C}$']
    rewards_names = [r'Dipole moment$(\mu)$',r'Isotropic polarizability$(\alpha)$',r'Highest occupied molecular orbital energy$(\epsilon_{HOMO})$',
    r'Lowest unoccupied molecular orbital energy$(\epsilon_{LUMO})$',r'Gap Between $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$$(\Delta \epsilon)$',
    r'Electronic spatial extent$(\langle R^2 \rangle)$',r'Zero point vibrational energy(ZPVE)',r'Internal energy at 0K$(U_0)$',r'Internal energy at 298.15K$(U)$',
    r'Enthalpy at 298.15K$(H)$',r'Free energy at 298.15K$(G)$',r'Heat capacity at 298.15K$(c_{v})$',r'Atomization energy at 0K$(U_0^{ATOM})$',
    r'Atomization energy at 298.15K$(U^{ATOM})$',r'Atomization enthalpy at 298.15K$(H^{ATOM})$',r'Atomization free energy at 298.15K$(G^{ATOM})$',
    r'Rotational constant A',r'Rotational constant B',r'Rotational constant C']
    #print(reward)
    axes_run.set_title(r'$QM9:$'+rewards_names[configs['reward']],)#fontsize=20)
    axes_run.set(xlabel=r'Number of Batches', ylabel=r"Runtime per Batch(log(s))")
    col_counter += 1


    lines_labels = axes_run.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    #print([tuple(lines)])
    fig_run.legend(lines,labels, loc ='lower center',bbox_to_anchor=(0.6, -0.12), fancybox = True, ncol = 5,prop = { "size": 8 },)
    #fig_rew.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1),  prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})


    #fig.tight_layout()
    #plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    print('here3')
    fig_run.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_runtimes.pdf',bbox_inches='tight')
    fig_run.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_runtimes.svg',bbox_inches='tight',format='svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--base_dir', type=str,  default='/cluster/scratch/bsoyuer/base_code/graph_BO/results/')
    parser.add_argument('--exp_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_name', type=str,  default='default_vs_random_rew0')

    args = parser.parse_args()
    main(args)