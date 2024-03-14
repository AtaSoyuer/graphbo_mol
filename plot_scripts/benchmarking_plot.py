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
    'num_actions': 130700, # any number below 10000 works.
    # GNN-UCB
    'GD_batch_size':50,
    'T':1501,
    'T0':80, 
    'T1':50, 
    'T2':400,
    'alg_lambda': 0.003,
    'alpha':2.5,
    'batch_size':50,
    'batch_window_size':10,
    'dim':64,
    'dropout_prob':0.2,
    'exploration_coef': 0.5,
    'explore_threshold': 10,
    'factor':0.7,
    'gamma':2.0,
    'net': 'GNN',
    'lr': 1e-3,
    'neuron_per_layer':128,
    'num_mlp_layers_alg': 1,
    'pretrain_steps': 100,
    'pretrain_indices_name': 'pretrain_indices',
    'pretrain_model_name': 'nnconv_reward3_5000samples_100ep',
    'pool_num':200,
    'print_every':250,
    'patience':5,
    'stop_count' : 9000,
    'small_loss' : 1e-3,
    'subsample_method':'random', 
    'subsample_num':20, 
    'synthetic':0,
    'relative_improvement' : 1e-6,
    'reward': 3,
    'reward_plot_dir':3,
    'select_K':5,
    #BOOLEAN ARGS:
    'no_var_computation':'false',
    'alternative':'true',
    'batch_GD':'true',
    'pool':'true',
    'load_pretrained':'true',
    'large_scale':'true',
    'ucb_wo_replacement':'true',
    'focal_loss':'false',
    'pool_top_means':'false',
    'batch_window':'false',
    'select_K_together':'false',
    'laplacian_features': 'false',
    'pretraining_load_pretrained': 'false',
    'remove_smiles_for_sober': 'true',
    'runner_verbose': 'false',
    'thompson_sampling': 'false',
}


def main(args):
    top_k_percentages = [0.01, 0.1, 1, 5, 10, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_sharpness = [0.5, 0.95]
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
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_reg, axes_reg = plt.subplots( ncols = 1, nrows=1, (12,9))
    fig_reg, axes_reg = plt.figure(), plt.axes()

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_topk, axes_topk = plt.subplots( ncols = 1, nrows=1, )
    fig_topk, axes_topk = plt.figure(), plt.axes()

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_cal, axes_cal = plt.subplots( ncols = 1, nrows=1, )
    fig_cal, axes_cal = plt.figure(), plt.axes()

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_sha, axes_sha = plt.subplots( ncols = 1, nrows=1, )
    fig_sha, axes_sha = plt.figure(), plt.axes()
    #print(len(axes))

    #plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
    #plt.rcParams.update(bd.icml2022())
    plt.rcParams.update(bundles.icml2022(ncols=1,nrows=1,tight_layout=True))
    #fig_sha, axes_sha = plt.subplots( ncols = 1, nrows=1, )
    fig_rew, axes_rew = plt.figure(), plt.axes()
    #print(len(axes))

    row_counter = 0
    col_counter =0
    for net in ['GNN']:
        counter = 0
        df_net = df.loc[df['net'] == net]
        #print(df_net)
        configurations = [config for config in
                        zip(
                        # df_net['alg_lambda'], 
                        # df_net['exploration_coef'], 
                        # df_net['pretrain_steps'], 
                        # df_net['neuron_per_layer'], 
                        # df_net['lr'], 
                        # df_net['stop_count'], 
                        df_net['T'], 
                        # df_net['small_loss'], 
                        df_net['reward'],
                        # df_net['GD_batch_size'], 
                        df_net['T2'], 
                        # df_net['dim'], 
                        # df_net['gamma'], 
                        # df_net['alpha'],
                        # df_net['pool_num'], 
                        # df_net['batch_size'], 
                        # df_net['num_actions'], 
                        # df_net['alternative'],
                        # df_net['batch_GD'], 
                        # df_net['pool'], 
                        # df_net['load_pretrained'], 
                        # df_net['large_scale'], 
                        # df_net['ucb_wo_replacement'], 
                        # df_net['focal_loss'], 
                        # df_net['pool_top_means'],
                        # df_net['batch_window_size'],
                        # df_net['batch_window'],
                        df_net['no_var_computation'],
                        #df_net['pretrain_model_name'],
                        df_net['oracle'],
                        ) if
                        not all(z == config[0] for z in config[1:])]
        configurations = list(set(configurations))
        #print(len(configurations))
        # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
        for config in configurations:
            print(len(configurations))
            # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
            # alg_lambda = config[0]
            # exp_coef = config[1]
            # pretrain_steps = config[2]
            # neuron_per_layer = config[3]
            # lr = config[4]
            # stop_count = config[5]
            T = config[0]
            #small_loss = config[7]
            reward = config[1]
            #GD_batch_size = config[9]
            T2 = config[2]
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
            no_var_computation = config[3]
            oracle = config[4]
            #pretrain_model_name = config[4]

            #sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
            # sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]
            # sub_df = sub_df.loc[sub_df['neuron_per_layer'] == neuron_per_layer]
            # sub_df = sub_df.loc[sub_df['lr'] == lr] 
            # sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
            # sub_df = sub_df.loc[sub_df['stop_count'] == stop_count] 
            #sub_df = sub_df.loc[sub_df['T'] == T] 
            # sub_df = sub_df.loc[sub_df['small_loss'] == small_loss]
            #sub_df = sub_df.loc[sub_df['reward'] == reward]
            # sub_df = sub_df.loc[sub_df['GD_batch_size'] == GD_batch_size]
            sub_df = df_net.loc[df_net['T2'] == T2]
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
            sub_df = sub_df.loc[sub_df['no_var_computation'] == no_var_computation]
            sub_df = sub_df.loc[sub_df['oracle'] == oracle]
            #sub_df = sub_df.loc[sub_df['pretrain_model_name'] == pretrain_model_name]


            # curve_name = r'$- \beta = $' + '{:.7f}'.format(exp_coef) + r'$- width = $' + '{:.5f}'.format(neuron_per_layer) +'\n' \
            #     + r'$- lr = $' + '{:.6f}'.format(lr)  + r'$- T = $' + '{:.5f}'.format(T) + r'$- stop_count = $' + '{:.5f}'.format(stop_count) +'\n' + \
            #     r'$- T2 = $' + '{:.7f}'.format(T2) + r'$- pool_size = $' + '{:.7f}'.format(pool_num) + r'$- batch_size = $' + '{:.7f}'.format(batch_size) + '\n' +\
            #     r'$- alternative = $' + '{:.7f}'.format(alternative) + r'$- batch_GD = $' + '{:.7f}'.format(batch_GD) + r'$- pool = $' + '{:.7f}'.format(pool) + '\n' +\
            #     r'$- load_pretrained = $' + '{:.7f}'.format(load_pretrained) + r'$- ucb_wo_replacement = $' + '{:.7f}'.format(ucb_wo_replacement) + r'$- focal_loss = $' + '{:.7f}'.format(focal_loss) + '\n' \
            #     + r'$- pool_top_means = $' + '{:.7f}'.format(pool_top_means) + r'$- batch_window = $' + '{:.7f}'.format(batch_window) + r'$- batch_window_size = $' + '{:.7f}'.format(batch_window_size) + r'$- no_var_computation = $' + '{:.7f}'.format(no_var_computation) \

            curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{GNN-SS-GREEDY} $', r'$\bf{GNN-SS-RANDOM} $', r'$\bf{GNN-SS-ORACLE} $']
            #curve_name_prepends = [r'$\bf{GNN-SS-UCB} $', r'$\bf{GNN-SS-UCB(TRANSFER)} $', r'$\bf{GNN-SS-RANDOM} $', r'$\bf{GNN-SS-ORACLE} $']
            # curve_name_params = r'$\bf{- \beta =} $' + '{:.3f}'.format(exp_coef) + r'$\bf{- \lambda =} $' + '{:.3f}'.format(alg_lambda) + r'$\bf{- T_{warmup} =} $' + '{:.1f}'.format(T2) + r'$\bf{- T = }$' + '{:.1f}'.format(T-1) + r'$\bf{- GCN\_width =} $' + '{:.1f}'.format(256.0) + '\n' \
            # curve_name_params = r'$\bf{- \beta =} $' + '{:.3f}'.format(exp_coef) + r'$\bf{- \lambda =} $' + '{:.3f}'.format(alg_lambda) + r'$\bf{- T_{warmup} =} $' + '{:.1f}'.format(T2) + r'$\bf{- T = }$' + '{:.1f}'.format(T-1) + r'$\bf{- GCN\_width =} $' + '{:.1f}'.format(256.0) + '\n' \
            #      + r'$\bf{- lr =}$' + '{:.4f}'.format(lr) + r'$\bf{- GD\_stop\_count =} $' + '{:.1f}'.format(stop_count) + r'$\bf{- pool\_size =} $' + '{:.1f}'.format(pool_num) + r'$\bf{- batch\_size =} $' + '{:.1f}'.format(batch_size)
        
            #print(sub_df['pretrain_model_name'])
            #if (sub_df['pretrain_model_name'] == 'nnconv_reward3n4_8000samples_100ep').all():
            if (sub_df['no_var_computation'] == True).all():
                curve_name = curve_name_prepends[1] #+ curve_name_params
                print(curve_name)
                clr = generic_lines[1]
                lst = linestyles[0]
            elif (sub_df['oracle'] == True).all():
                curve_name = curve_name_prepends[3] #+ curve_name_params
                print(curve_name)
                clr = generic_lines[3]
                lst = linestyles[3]
            elif (sub_df['T2'] == sub_df['T']).all():
                curve_name = curve_name_prepends[2] #+ curve_name_params
                print(curve_name)
                clr = generic_lines[2]
                lst = linestyles[3]
            else:
                curve_name = curve_name_prepends[0] #+ curve_name_params
                print(curve_name)
                clr = generic_lines[0]
                lst = linestyles[0]
                    
            # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
            regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
            top_k = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['top_k']])
            coverages = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['coverages']])
            avg_widhts = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['avg_widths']])
            collected_max = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['collected_max']])

            rewards_all = np.array([np.squeeze(np.array(percentages)) for percentages in sub_df['rewards']])
            print('REWS ALL:', rewards_all.shape)

            simple_regret = np.zeros((rewards_all.shape[0], rewards_all.shape[1]-1))
            print('SIMPLE_REGRET:', simple_regret.shape)
            print('SHAPE:', np.array([np.max(rewards_all[0,i]).item() for i in range(1,rewards_all.shape[1])]).shape)
            #print('VALUES:', np.array([np.max(rewards_all[0,i]).item() for i in range(1,rewards_all.shape[1])])[:20])

            for row in range(rewards_all.shape[0]):
                simple_regret[row,:] = np.array([np.max(rewards_all[row,:i]) for i in range(1,rewards_all.shape[1])])


            axes_reg.plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
            axes_reg.fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.4*np.std(regrets_bp, axis=0),
                                np.mean(regrets_bp, axis=0)+0.4*np.std(regrets_bp, axis=0), alpha=0.2,
                                color = clr)
            axes_reg.grid(alpha=0.8)

            axes_rew.plot(np.mean(simple_regret, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst)
            axes_rew.fill_between(np.arange(configs['T']-1), np.mean(simple_regret, axis=0)-0.8*np.std(simple_regret, axis=0),
                                np.mean(simple_regret, axis=0)+0.8*np.std(simple_regret, axis=0), alpha=0.2,
                                color = clr)
            axes_rew.grid(alpha=0.8)
            
            axes_topk.plot(top_k_percentages , np.mean(top_k, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = lst, marker='o', markersize=4)
            #axes[1,0].fill_between(np.arange(configs['T']), np.mean(top_k, axis=0)-0.2*np.std(top_k, axis=0),
                                #np.mean(top_k, axis=0)+0.2*np.std(top_k, axis=0), alpha=0.2,
                                #color = generic_lines[counter])
            axes_topk.errorbar(top_k_percentages, np.mean(top_k, axis=0), yerr=0.4*np.std(top_k, axis=0), fmt='o', markersize=5, capsize=6, color = clr)
            axes_topk.set_xscale('log')
            axes_topk.grid(alpha=0.8)

            if (sub_df['no_var_computation'] == False).all():
                #print('True')
                axes_cal.plot(alphas, np.mean(coverages, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = clr, linestyle = linestyles[counter%13])
                axes_cal.fill_between(alphas, np.mean(coverages, axis=0)-np.std(coverages, axis=0),
                                    np.mean(coverages, axis=0)+np.std(coverages, axis=0), alpha=0.2,
                                    color = clr)
                axes_cal.plot([0.0,1.0], [0.0,1.0], color='black', linewidth=0.5, alpha=0.1)
                axes_cal.grid(alpha=0.8)
                #print(alpha_sharpness_str)
                axes_sha.bar(alpha_sharpness_str , np.mean(avg_widhts, axis=0), label = curve_name, color = clr,  width=0.3, alpha=0.3)
                axes_sha.errorbar(alpha_sharpness_str, np.mean(avg_widhts, axis=0), yerr=0.4*np.std(avg_widhts, axis=0), fmt='o', markersize=4, capsize=8, color = clr)
            #print(collected_max)

            print(f'Collected global max {np.sum(collected_max)} out of {len(collected_max)} times, ratio={np.sum(collected_max)/len(collected_max)}')
            print(f'TopK Means: {np.mean(top_k, axis=0)}')
            print(f'TopK Stds: {np.std(top_k, axis=0)}')

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
        axes_reg.set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
        axes_rew.set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
        axes_topk.set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
        axes_cal.set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)
        axes_sha.set_title(r'$QM9:$'+rewards_names[reward],)#fontsize=20)

        axes_reg.set(xlabel=r'Online Oracle Calls', ylabel=r'$R_t$')
        axes_rew.set(xlabel=r'Online Oracle Calls', ylabel=r'$top\_1$')
        axes_topk.set(xlabel=r'Top-K Percent Samples Evaluated', ylabel=r"Ratio of Top-K Samples Acquired")
        axes_cal.set(xlabel=r"$\alpha$", ylabel=r'Calibration of Confidence Intervals')
        axes_sha.set(xlabel=r"$\alpha$", ylabel=r'Sharpness of Confidence Intervals')
        col_counter += 1

    print('here1')
    #lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[0][1],axes[1][0],axes[1][1]]]
    lines_labels = axes_reg.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    #print([tuple(lines)])
    fig_reg.legend(lines,labels, loc ='upper left',bbox_to_anchor=(0.17, 0.9), fancybox = True, ncol = 3, prop = { "size": 8 },)
    #fig_reg.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1),  prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    lines_labels = axes_rew.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    #print([tuple(lines)])
    fig_rew.legend(lines,labels, loc ='upper left',bbox_to_anchor=(0.17, 0.9), fancybox = True, ncol = 3,prop = { "size": 8 },)
    #fig_rew.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1),  prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    lines_labels = axes_topk.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    fig_topk.legend(lines,labels, loc ='upper left',bbox_to_anchor=(0.125, 0.9), fancybox = True, ncol = 3,prop = { "size": 8 },)
    #fig_topk.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1), prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    lines_labels = axes_cal.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    fig_cal.legend(lines,labels, loc ='lower right',bbox_to_anchor=(1.0, 0.125), fancybox = True, ncol = 2,prop = { "size": 8 },)
    #fig_cal.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1), prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    lines_labels = axes_sha.get_legend_handles_labels()
    #ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines, labels = lines_labels
    fig_sha.legend(lines,labels, loc ='upper left',bbox_to_anchor=(0.125, 0.9), fancybox = True, ncol = 3, prop = { "size": 8 },)
    #fig_sha.legend([tuple(lines)], [curve_name_params], loc ='lower center', bbox_to_anchor=(0.5, -0.1), prop = { "size": 5 }, fancybox = True, ncol = 2, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

    #fig.tight_layout()
    #plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
    if os.path.exists(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}'):
        pass
    else:
        os.makedirs(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}')
    print('here3')
    fig_reg.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_reg.pdf',bbox_inches='tight')
    fig_rew.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew.pdf',bbox_inches='tight')
    print('here4')
    fig_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_topk.pdf',bbox_inches='tight')
    fig_cal.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_cal.pdf',bbox_inches='tight')
    fig_sha.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_sha.pdf',bbox_inches='tight')
    fig_reg.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_reg.svg',bbox_inches='tight',format='svg')
    fig_topk.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_topk.svg',bbox_inches='tight',format='svg')
    fig_cal.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_cal.svg',bbox_inches='tight',format='svg')
    fig_sha.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_sha.svg',bbox_inches='tight',format='svg')
    fig_rew.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/{args.plot_dir}/{plot_name}_rew.svg',bbox_inches='tight',format='svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regret Run')
    # experiment parameters
    parser.add_argument('--base_dir', type=str,  default='/cluster/scratch/bsoyuer/base_code/graph_BO/results/')
    parser.add_argument('--exp_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_dir', type=str,  default='hyperparamgnnucb_bartu_10')
    parser.add_argument('--plot_name', type=str,  default='default_vs_random_rew0')

    args = parser.parse_args()
    main(args)