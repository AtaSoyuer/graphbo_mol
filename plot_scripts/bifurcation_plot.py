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
import umap
from matplotlib import cm
import numpy as np
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
    'reward':1,
    'reward_plot_dir':0,
}


def main(args):

    bund = bd.icml2022()
    bund['text.usetex'] = False
    bund['font.family'] = 'DejaVu Serif'
    #print('Bund:', bund)



    #plot_name = 'new_paper_hyperparam_{}T_{}actions_QM9'.format(configs['T'], configs['num_actions'])
    plot_name = args.plot_name

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_reg, axes_reg = plt.subplots( ncols = 1, nrows=1, (12,9))
    fig_run, axes_run = plt.figure(), plt.axes()

    row_counter = 0
    col_counter =0
    counter = 0
    #print(df_net)
    df_full, _ = collect_exp_results(exp_name=args.exp_dir)
    print(df_full['alternative'])
    print(df_full['thompson_sampling'])
    print(df_full['reward'])
    #df_new = df_full.loc[df_full['feat_dim'] == configs['feat_dim']]
    df_net = df_full
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
                    #df_net['no_var_computation'],
                    #df_net['oracle'],
                    #df_net['rand'],
                    df_net['thompson_sampling'],
                    #df_net['algorithm'],
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
        #no_var_computation = config[2]
        #oracle = config[3]
        #rand = config[4]
        thompson_sampling = config[2]
        #algorithm = config[4]
        #pretrain_model_name = config[4]
        print(reward)
        print(alternative)
        print(thompson_sampling)

    #     #sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
    #     # sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]
    #     # sub_df = sub_df.loc[sub_df['neuron_per_layer'] == neuron_per_layer]
    #     # sub_df = sub_df.loc[sub_df['lr'] == lr] 
    #     # sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
    #     # sub_df = sub_df.loc[sub_df['stop_count'] == stop_count] 
    #     # sub_df = sub_df.loc[sub_df['T'] == T] 
    #     # sub_df = sub_df.loc[sub_df['small_loss'] == small_loss]
    #     #sub_df = df_net.loc[df_net['reward'] == reward]
    #     # sub_df = sub_df.loc[sub_df['GD_batch_size'] == GD_batch_size]
    #     #sub_df = df_net.loc[df_net['T2'] == T2]
    #     # sub_df = sub_df.loc[sub_df['dim'] == dim] 
    #     # sub_df = sub_df.loc[sub_df['gamma'] == gamma] 
    #     # sub_df = sub_df.loc[sub_df['alpha'] == alpha] 
    #     # sub_df = sub_df.loc[sub_df['pool_num'] == pool_num]
    #     # sub_df = sub_df.loc[sub_df['batch_size'] == batch_size] 
    #     # sub_df = sub_df.loc[sub_df['num_actions'] == num_actions]
    #     # sub_df = sub_df.loc[sub_df['alternative'] == alternative]
    #     # sub_df = sub_df.loc[sub_df['batch_GD'] == batch_GD]
    #     # sub_df = sub_df.loc[sub_df['pool'] == pool]
    #     # sub_df = sub_df.loc[sub_df['load_pretrained'] == load_pretrained]
    #     # sub_df = sub_df.loc[sub_df['large_scale'] == large_scale]
    #     # sub_df = sub_df.loc[sub_df['ucb_wo_replacement'] == ucb_wo_replacement]
    #     # sub_df = sub_df.loc[sub_df['focal_loss'] == focal_loss]
    #     # sub_df = sub_df.loc[sub_df['pool_top_means'] == pool_top_means]
    #     # sub_df = sub_df.loc[sub_df['batch_window_size'] == batch_window_size]
    #     # sub_df = sub_df.loc[sub_df['batch_window'] == batch_window]
    #     # sub_df = df_net.loc[df_net['algorithm'] == algorithm]
    #     # sub_df = sub_df.loc[sub_df['no_var_computation'] == no_var_computation]
    #     # sub_df = sub_df.loc[sub_df['oracle'] == False]
    #     # sub_df = sub_df.loc[sub_df['rand'] == False]
    #     # sub_df = sub_df.loc[sub_df['thompson_sampling'] == thompson_sampling]
    #     #sub_df = sub_df.loc[sub_df['pretrain_model_name'] == pretrain_model_name]
    #     print('df_net:', df_net)
    #     sub_df = df_net.loc[df_net['rewards'] == reward]
    #     print('subdf:', sub_df)
    #     sub_df = sub_df.loc[sub_df['alternative'] == alternative]
    #     print('subdf:', sub_df)
    #     sub_df = sub_df.loc[sub_df['thompson_sampling'] == thompson_sampling]
    #     print('subdf:', sub_df)
        
        #sub_df = df_net.loc[df_net['reward'] == reward]
        sub_df = df_net

        curve_name_prepends = [r'$\bf{GNN-SS-UCB} $']
    
        #print(sub_df['pretrain_model_name'])
        #if (sub_df['pretrain_model_name'] == 'nnconv_reward3n4_8000samples_100ep').all():
        
        curve_name = curve_name_prepends[0] #+ curve_name_params
        print(curve_name)
        clr = generic_lines[0]
        print(clr)
        lst = linestyles[0]
        
        dataset = QM9(root="../../data/QM9")
        import csv

        TARGET = np.unique(reward)
        print('TARGET:',TARGET)

        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        mean, std = mean[:, TARGET].item(), std[:, TARGET].item()

        indices_to_remove_file = open("/cluster/scratch/bsoyuer/base_code/SOBER/indices_of_smiles_toremove_github.csv", "r")
        indices_to_remove = list(csv.reader(indices_to_remove_file, delimiter=","))[0]
        indices_to_remove_file.close()
        indices_to_remove = [int(item) for item in indices_to_remove]

        rows_to_keep = [x for x in range(dataset.data.y.shape[0]) if x not in indices_to_remove]
        #print(rows_to_keep[-1])
        # dataset2 = QM9(path, transform=MyTransform(target))
        # print('original max:', torch.max(dataset2.data.y[rows_to_keep,target]))

        remaining_rewards = dataset.data.y[rows_to_keep,TARGET]
        print('remaining_rewards:',remaining_rewards.shape)
        MAX_REWARD = np.max(np.array(dataset.data.y)[rows_to_keep,TARGET])

        dataset_removed = dataset[rows_to_keep]
        MAX_NUM_NODES = 29

        index = rows_to_keep
        graph_rewards = [d.y[0,TARGET] for d in dataset_removed]
        print('graph rewards:', graph_rewards[0])
        top_idx = np.argsort(np.array(graph_rewards))[-10:]
        print(graph_rewards[top_idx[0]])
        print('top 10:', [graph_rewards[i] for i in top_idx])

        def feat_pad(feat_mat):
            return torch.nn.functional.pad(feat_mat,pad=(0,0,0,MAX_NUM_NODES-len(feat_mat)), value=0)#value=float('nan'))
        
        def z_pad(feat_mat):
            return torch.nn.functional.pad(feat_mat,pad=(0,MAX_NUM_NODES-len(feat_mat)), value=0)# value=float('nan'))   
        
        def rand_jitter(arr):
            stdev = .001 * (max(arr) - min(arr))
            return arr + np.random.randn(len(arr)) * stdev

        features_list = []
        rewards_list = []
        #print('len dataset:', len(dataset))
        count = 0
        for ix in index:
            #print(ix)
            #features_list.append(feat_pad(torch.tensor(graph_data[ix].x)).flatten())
            features_list.append(torch.cat((feat_pad(dataset[ix].x.float()), feat_pad(dataset[ix].pos.float()), z_pad(dataset[ix].z.float())[:,None]), 1).flatten())
            rewards_list.append(graph_rewards[count]) 
            count += 1

        rewards_arr = np.array(rewards_list)
        rewards_arr.resize((len(dataset),1))

        #for e in features_list:
            #print(e.shape)
        
        features_list = torch.stack(features_list).numpy().astype(np.float32)
        print('features_list:', features_list)

        #print("Features_list_shape:", features_list.shape)
        #print('Type Features List:', type(features_list))

        #reducer = umap.UMAP(n_neighbors=int(NUM_ACTIONS/10), min_dist=0.2)
        reducer = umap.UMAP(n_neighbors=80, min_dist=0.05, n_components=1, random_state=3782)
        embedding = reducer.fit_transform(features_list)
    
                
        actions = np.array([np.squeeze(np.array(item)) for item in sub_df['actions']])[0]
        print('actions:', actions)
        rewards_all = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['rewards']])[0]*std+mean
        print('rewards:', rewards_all)

        #simple_regret = np.zeros((rewards_all.shape[0], rewards_all.shape[1]-1))
        #for row in range(rewards_all.shape[0]):
            #simple_regret[row,:] = np.array([np.max(rewards_all[row,:i]) for i in range(1,rewards_all.shape[1])])

        bifurcation = np.zeros((actions.shape[0], 2))
        for i in range(actions.shape[0]):
            print(embedding[int(actions[i])])
            bifurcation[i,0] = embedding[int(actions[i])]
            bifurcation[i,1] = i

        print(bifurcation)
        

        pcm1 = axes_run.scatter(
                bifurcation[:, 0],
                bifurcation[:, 1],
                c=rewards_all,
                cmap='plasma_r',
                marker = "x",
                s = 0.2,
                alpha=0.5)

        cmap = cm.get_cmap("plasma_r")
        fig_run.colorbar(pcm1, ax=axes_run)

        count = 10
        for i in top_idx:
            #axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
            #axes_run.text(embedding[i],-50,f'Top-{count}',rotation=90, fontsize='xx-small')

            # if count not in [2,5,6,9]:
            #     axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
            #     axes_run.text(embedding[i],-50,'Top-1,3,4,7,8,9,10',rotation=90, fontsize='xx-small')
            # if count in [2,5,6,9]:
            #     axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
            #     axes_run.text(embedding[i],-50,f'Top-{count}',rotation=90, fontsize='xx-small')

            if count in [1,7]:
                axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
                axes_run.text(embedding[i],-50,'Top-1,7',rotation=90, fontsize='xx-small')
            elif count in [3,6]:
                axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
                axes_run.text(embedding[top_idx[7]],-50,'Top-3,6',rotation=90, fontsize='xx-small')
            elif count in [2,5,9,10]:
                axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
                axes_run.text(embedding[top_idx[8]],-50,'Top-2,5,9,10',rotation=90, fontsize='xx-small')
            else:
                axes_run.axvline(x = embedding[i], color = 'dimgray', linestyle = 'dashed', linewidth=0.5)
                axes_run.text(embedding[i],-50,f'Top-{count}',rotation=90, fontsize='xx-small')
            count -= 1

        counter += 1
            
    rewards_names = [r'Dipole moment$(\mu)$',r'Isotropic polarizability$(\alpha)$',r'Highest occupied molecular orbital energy$(\epsilon_{HOMO})$',
    r'Lowest unoccupied molecular orbital energy$(\epsilon_{LUMO})$',r'Gap Between $\epsilon_{HOMO}$ and $\epsilon_{LUMO}$$(\Delta \epsilon)$',
    r'Electronic spatial extent$(\langle R^2 \rangle)$',r'Zero point vibrational energy(ZPVE)',r'Internal energy at 0K$(U_0)$',r'Internal energy at 298.15K$(U)$',
    r'Enthalpy at 298.15K$(H)$',r'Free energy at 298.15K$(G)$',r'Heat capacity at 298.15K$(c_{v})$',r'Atomization energy at 0K$(U_0^{ATOM})$',
    r'Atomization energy at 298.15K$(U^{ATOM})$',r'Atomization enthalpy at 298.15K$(H^{ATOM})$',r'Atomization free energy at 298.15K$(G^{ATOM})$',
    r'Rotational constant A',r'Rotational constant B',r'Rotational constant C']

    axes_run.set_title(r'$QM9:$'+rewards_names[configs['reward']],)#fontsize=20)
    axes_run.set(xlabel=r'1D UMAP Projection of QM9 Dataset', ylabel=r"Time-Steps(t)")
    col_counter += 1


    # lines_labels = axes_run.get_legend_handles_labels()
    # #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # lines, labels = lines_labels
    # #print([tuple(lines)])
    # fig_run.legend(lines,labels, loc ='lower center',bbox_to_anchor=(0.6, -0.14), fancybox = True, ncol = 6,prop = { "size": 10 },)
    # #fig_rew.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1),  prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})


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