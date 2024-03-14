import torch_geometric
from torch_geometric.datasets import QM9
import os
import json
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch


dataset = QM9(root="../data/QM9")

MAX_NUM_NODES = 29

BATCH_SIZE = 25 

TARGET = 3

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


env_rds = np.random.RandomState(354)

subset_indices = np.random.choice(len(dataset), 128000, replace=False)
print('LEN:',np.unique(subset_indices).shape)

#dataset_new = dataset_new.shuffle()
train_dataset = dataset[subset_indices] #REMOVE SHUFFLES AND USE OLD RANDOMLY SELECTED INDICES

print('UNIQUE REWS:',np.unique(np.append(np.array(train_dataset.data.idx)[subset_indices],1000000)).shape)

print('UNIQUE REWS:',np.unique(np.array(train_dataset.data.idx)[subset_indices]).shape)

print('UNIQUE REWS:',np.unique(np.append(np.array(train_dataset.data.idx)[subset_indices],train_dataset.data.idx[0])).shape)

print(np.array(train_dataset.data.idx)[subset_indices])

print('Max rew:', np.max(np.array(train_dataset.data.y)[:,TARGET]))

# indices = [3,5,7]

# print('CHOSEN REWS:',np.array([train_dataset[i].y.reshape(-1,1)[TARGET] for i in indices]))
# print('CHOSEN REWS2:',np.array(train_dataset.data.y)[indices,TARGET])

# mean = dataset.data.y.mean(dim=0, keepdim=True)
# std = dataset.data.y.std(dim=0, keepdim=True)
# mean, std = mean[:, TARGET].item(), std[:, TARGET].item()
# dataset.data.y = (dataset.data.y[:,TARGET] - mean) / std

# print('UNIQUE REWS ORG:',np.unique(np.array(dataset.data.y).round(decimals=20)).shape)
# print('UNIQUE REWS TRAIN:',np.unique(np.array(dataset.data.y)[subset_indices].round(decimals=20)).shape)
# print('ORG LEN:', print(len(dataset)))

# print(mean)
# print(std)

# graph_rewards = [d.y for d in dataset]

# plt.figure()
# #plt.errorbar(torch.tensor(means).flatten(), torch.tensor(graph_rewards).flatten(), xerr=conf_bounds, fmt='o', alpha=0.2)
# plt.scatter(torch.tensor(graph_rewards).flatten(), torch.tensor(graph_rewards).flatten(), s=1/2)
# plt.grid(alpha=0.3)
# plt.title(r'$Seen\ Samples\ vs\ Unseen\ Samples$')
# plt.xlabel(r'$Predicted$')
# plt.ylabel(r"$True\ Reward$")
# plt.savefig('control.png')







# path = '/cluster/scratch/bsoyuer/base_code/graph_BO/results/rew3n4_transfer/6331689487753742719/'
# # Opening JSON file
# def open_json(path):
#     with open(path) as json_file:
#         data = json.load(json_file)
#         return data

# dicts_list = [open_json(path+d) for d in os.listdir(path) if d.endswith('.json')]

# #print(len(dicts_list))

# mean = train_dataset.data.y.mean(dim=0, keepdim=True)
# std = train_dataset.data.y.std(dim=0, keepdim=True)
# mean, std = mean[:, TARGET].item(), std[:, TARGET].item()
# print('mean:', mean)
# print('std:', std)
# print('original_max:',np.max(np.array(dataset.data.y[:,TARGET])))
# indices = np.argpartition(np.array(dataset.data.y[:,TARGET]), -10)[-10:]
# print('original_maximums:',np.array(dataset.data.y)[indices,TARGET])

# maximums = []
# maximums2 = []
# for d in dicts_list:
#     collected_rews = np.array(d['exp_results']['rewards']).flatten()
#     collected_rews = collected_rews*std+mean
#     print('Max:',np.max(collected_rews))
#     ind = np.argpartition(collected_rews, -10)[-10:]
#     maximums.append(collected_rews[ind])

#     # collected_inds = np.array(d['exp_results']['actions'])
#     # print('shape:',collected_inds.shape)
#     # collected_rewards = np.array(train_dataset.data.y[[collected_inds],TARGET])
#     # print('shape:',collected_rewards.shape)
#     # #ind = np.argpartition(collected_rewards, -10)[-10:]
#     # #maximums2.append(collected_rewards[ind])
#     # print('MAXNEW:', np.max(collected_rewards))

# print(maximums)
# print(np.mean(np.array(maximums)))
# print(np.max(np.array(maximums)))

    





from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

rewards_all = []

for r in range(19):
    rewards = []
    for i, data in enumerate(loader):
        # Every data instance is an input + label pair
        reward = data.y[:,r].flatten().tolist()
        rewards.extend(reward)
    rewards_all.append(rewards)


rewards_names = [r'$\mu$',r'$\alpha$',r'$H\epsilon_{HOMO}$',
r'$\epsilon_{LUMO}$',r'$\Delta \epsilon_{HOMO-LUMO}$',
r'$\langle R^2 \rangle$',r'$ZPVE$',r'$U_0$',r'$U$',
r'$H$',r'$G$',r'$c_{v}$',r'$U_0^{ATOM}$',
r'$U^{ATOM}$',r'$H^{ATOM}$',r'$G^{ATOM}$',
r'$A$',r'$B$',r'$C$']

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import pandas as pd
import subprocess
import os


corr_mat = np.zeros((19,19))

for i in range(19):
    for j in range(19):
        corr_mat[i,j] = pearsonr(rewards_all[i], rewards_all[j])[0]

plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
#sns.set(font_scale=0.4)
hm = sns.heatmap(corr_mat,vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            #cmap = 'coolwarm',
            annot=False,
            annot_kws={"size": 30},
            xticklabels = rewards_names,
            yticklabels = rewards_names,)

hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.title(r'$PearsonR\ Correlation\ of\ QM9\ Rewards$')

plt.savefig('heatmap.svg', format='svg')
command = ['svg42pdf', os.path.join(os.getcwd(), 'heatmap.svg'), 'heatmap.pdf']
subprocess.run(command, stderr=subprocess.PIPE, text=True)