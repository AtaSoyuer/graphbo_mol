import os
import sys
sys.path.append('/cluster/scratch/bsoyuer/base_code/graph_BO')
from utils_exp import collect_exp_results
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager
from plot_specs import *
import bundles
#DIR = '/local/pkassraie/gnnucb/results/'
DIR='/cluster/scratch/bsoyuer/base_code/graph_BO/results/'
df_full, _ = collect_exp_results(exp_name='hyperparamgnnucb_bartu_8')

'''
CHANGE EXP_NAME IN THE ABOVE LINE TO READ FROM DIFFERENT DIRECTORY
WHEN READING DATA WITH NEW PARAMS, MAKE CHANGES IN LINES ~62, 73, 77, 79
ALSO, CAN READJUST GENERIC_LINES IN PLOT_SPECS.PY FOR BETTER LINE COLORS.
'''

configs = {
    # Dataset
    'num_nodes': 5, # or 20 or 100
    'edge_prob': 0.05, #or 0.2 or 0.95
    'feat_dim': 15, #10 or 100
    'num_actions': 150, # any number below 10000 works.
    # GNN-UCB
    'neuron_per_layer': 2048,
    'exploration_coef': 1e-3,
    'alg_lambda': 0.01,
    # other BO params
    'T' : 150,
    'dataset_size' : 130831,
    'lr': 1e-3
}

# pick which synthetic
#df = df_full.loc[df_full['num_nodes'] == configs['num_nodes']]
#df = df.loc[df['edge_prob'] == configs['edge_prob']]
#df = df.loc[df['num_actions']==configs['num_actions']]
#df = df.loc[df['feat_dim'] == configs['feat_dim']]
#df = df.loc[df['T'] == configs['T']]

#Pick which QM9
df = df_full.loc[df_full['dataset_size'] == configs['dataset_size']]
df = df.loc[df['num_actions']==configs['num_actions']]
df = df.loc[df['feat_dim'] == configs['feat_dim']]
df = df.loc[df['T'] == configs['T']]
#print(df)


plot_name = 'new_paper_hyperparam_{}T_{}N_{}d_{}p_{}actions'.format(configs['T'],configs['num_nodes'],configs['feat_dim'],configs['edge_prob'], configs['num_actions'])
plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
fig, axes = plt.subplots( ncols = 1, nrows=1)#, figsize = (8, 12)
plt.rcParams.update(bundles.neurips2022(ncols=1,nrows=1,  tight_layout=True))
#print(len(axes))

row_counter = 0
col_counter =0
for net in ['GNN']:
    counter = 0
    df_net = df.loc[df['net'] == net]
    #print(df_net)
    configurations = [config for config in
                      zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps'], df_net['neuron_per_layer'], df_net['lr'], df_net['stop_count'], df_net['relative_improvement'], df_net['small_loss'], df_net['num_mlp_layers_alg']) if
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
        relative_improvement = config[6]
        small_loss = config[7]
        num_mlp_layers_alg = config[8]
        sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
        sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
        sub_df = sub_df.loc[sub_df['neuron_per_layer'] == neuron_per_layer]
        sub_df = sub_df.loc[sub_df['lr'] == lr] 
        sub_df = sub_df.loc[sub_df['stop_count'] == stop_count] 
        sub_df = sub_df.loc[sub_df['relative_improvement'] == relative_improvement] 
        sub_df = sub_df.loc[sub_df['small_loss'] == small_loss]
        sub_df = sub_df.loc[sub_df['num_mlp_layers_alg'] == num_mlp_layers_alg] 

        curve_name = r'$\lambda = $' + '{:.7f}'.format(alg_lambda) + r'$- \beta = $' + '{:.7f}'.format(exp_coef) + r'$- width = $' + '{:.5f}'.format(neuron_per_layer) \
            + r'$- lr = $' + '{:.6f}'.format(lr)  + r'$- num_layers = $' + '{:.5f}'.format(num_mlp_layers_alg) + r'$- stop_count = $' + '{:.5f}'.format(stop_count) + \
                r'$- relative_improvement = $' + '{:.7f}'.format(relative_improvement) + r'$- small_loss = $' + '{:.7f}'.format(small_loss)
        # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
        regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
        # picked_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['pick_vars_all']])
        # avg_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['avg_vars']])
        #print('Axes:', axes[row_counter][col_counter])
        #print(np.mean(regrets_bp, axis=0))
        print(generic_lines[counter])
        axes.plot(np.mean(regrets_bp, axis=0), linestyle[net+'_UCB'],  label = curve_name, color = generic_lines[counter], linestyle = linestyles[counter%13])
        axes.fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0)-0.2*np.std(regrets_bp, axis=0),
                             np.mean(regrets_bp, axis=0)+0.2*np.std(regrets_bp, axis=0), alpha=0.2,
                             color = generic_lines[counter])

        counter += 1

    #axes[row_counter][col_counter].set_xlabel(r'$t$')
    axes.set_title(f'{net}-UCB')
    #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
    col_counter += 1

# DIR='/cluster/scratch/bsoyuer/base_code/graph_BO/results/'
# df_full, _ = collect_exp_results(exp_name='hyper_param_search_us_new')
# row_counter += 1
# col_counter = 0

# # # pick which synthetic
# #df = df_full.loc[df_full['num_nodes'] == configs['num_nodes']]
# #df = df.loc[df['edge_prob'] == configs['edge_prob']]
# #df = df.loc[df['num_actions']==configs['num_actions']]
# #df = df.loc[df['feat_dim'] == configs['feat_dim']]
# #df = df.loc[df['T'] == configs['T']]

# #Pick which QM9
# df = df_full.loc[df_full['dataset_size'] == configs['num_nodes']]
# df = df.loc[df['num_actions']==configs['num_actions']]
# df = df.loc[df['feat_dim'] == configs['feat_dim']]
# df = df.loc[df['T'] == configs['T']]

# for net in ['GNN']:
#     counter = 0
#     df_net = df.loc[df['net'] == net]
#     if net == 'NN':
#         df_net = df_net.loc[df_net['pretrain_steps']!= 40]
#     configurations = [config for config in
#                       zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if
#                       not all(z == config[0] for z in config[1:])]
#     configurations = list(set(configurations))
#         # for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'], df_net['exploration_coef'], df_net['pretrain_steps']) if not all():
#     for config in configurations:
#         #for alg_lambda, exp_coef, pretrain_steps in zip(df_net['alg_lambda'].unique(), df_net['exploration_coef'].unique(), df_net['pretrain_steps'].unique()):
#         alg_lambda = config[0]
#         exp_coef = config[1]
#         pretrain_steps = config[2]
#         sub_df = df_net.loc[df_net['alg_lambda'] == alg_lambda]
#         sub_df = sub_df.loc[sub_df['exploration_coef'] == exp_coef]
#         sub_df = sub_df.loc[sub_df['pretrain_steps'] == pretrain_steps]

#         curve_name = r'$\lambda = $' + '{:.5f}'.format(alg_lambda) + r'$- \beta = $' + '{:.5f}'.format(exp_coef)
#         # regrets = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets']])
#         regrets_bp = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['regrets_bp']])
#         # picked_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['pick_vars_all']])
#         # avg_vars = np.array([np.squeeze(np.array(regrets)) for regrets in sub_df['avg_vars']])
#         axes[row_counter][col_counter].plot(np.mean(regrets_bp, axis=0), linestyle[net+'_US'], label=curve_name)#, color=generic_lines_us[counter])
#         axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(regrets_bp, axis=0) - 0.2 * np.std(regrets_bp, axis=0),
#                                        np.mean(regrets_bp, axis=0) + 0.2 * np.std(regrets_bp, axis=0), alpha=0.2,)
#                                            #color=generic_lines_us[counter])
#         # axes[row_counter][col_counter].plot(np.mean(regrets, axis=0), linestyle[net], label=curve_name, color=generic_lines_gp[counter])
#         # axes[row_counter][col_counter].fill_between(np.arange(configs['T']), np.mean(regrets, axis=0) - np.std(regrets, axis=0),
#         #                         np.mean(regrets, axis=0) + np.std(regrets, axis=0), alpha=0.2,
#         #                         color=generic_lines_gp[counter])
#         counter += 1

#     #axes[row_counter][col_counter].set_xlabel(r'$t$')

#     axes[row_counter][col_counter].set_title(net+'-PE')
#     #axes[row_counter][col_counter].set_ylabel(r'$R_{BP,T}$')
#     col_counter += 1


#lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0][0], axes[0][1],axes[1][0],axes[1][1]]]
lines_labels = axes.get_legend_handles_labels()
#ines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
lines, labels = lines_labels
fig.legend(lines,labels, loc ='center',bbox_to_anchor=(0.5, -0.4), fancybox = True, ncol = 4)

#fig.tight_layout()
plt.rcParams.update(bundles.neurips2022(ncols=1, nrows = 1,  tight_layout=True))
plt.savefig(f'/cluster/scratch/bsoyuer/base_code/graph_BO/plots/hyperparamqm9/{plot_name}.pdf',bbox_inches='tight')

#plt.show(bbox_inches='tight')