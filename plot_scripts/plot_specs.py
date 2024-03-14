# mics for coherent plots

import numpy.random as random
import matplotlib

all_colors_icml = {
    'sharp_green': '#419d78',
    'sharp_red': '#d33f49',
    'candy_pink': '#e58b91',
    'sharp_yellow': '#f9a620',
    'candy_green': '#8ccfb4',
    'candy_yellow': '#fbc774',
    'blue': '#5171a5'
}

all_colors_genz = {
    'sharp_blue': '#575d90',
    'sharp_green': '#84a07c',
    'sharp_yellow': '#c3d350',
    'sharp_orange': '#dfab42',
    'sharp_red': '#d33f49',
    'sharp_iceberg': '#86AEDC',
    'soft_blue': '#8388a8',
    'soft_green': '#b0c398',
    'soft_yellow': '#d6dda7',
    'soft_red': '#e399a0',
    'soft_iceberg': '#C3D7EE',
    'soft_orange': '#EFD5A1'
}
#['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
#generic_lines = ['#D55E00', '#DE8F05', '#ECE133', '#029E73','#08cad1','#59adf6','#9d94ff','#2176AE']
#generic_lines = ['#ff6961', '#ffb480', '#F4ED66', '#42d6a4', '#08cad1', '#59adf6', '#9d94ff', '#c780e8', '#2176AE', '#ECE133', '#de8f05', '#0173b2', '#cc78bc', '#fbafe4', '#ff6961', '#56b4e9', '#EFD5A1', \
#'#C3D7EE', '#e399a0', '#d6dda7', '#b0c398', '#8388a8', '#c3d350', '#575d90', '#EB9605'] #,'#ff6961'
#get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
#generic_lines = get_colors(50)
#generic_lines =  [c for c in matplotlib.pyplot.cm.tab10.colors]
generic_lines = ['#59adf6', '#ffb480', '#ba5ced', '#ff6961', 'gray', '#006400', '#FF6FFF', '#FFFF00'] #UCB, Greedy, TS, Random, Orcale
generic_lines_us = ['#ff6961', '#ffb480', '#f8f38d', '#42d6a4', '#08cad1', '#59adf6', '#9d94ff', '#c780e8' , '#65655E', '#7D80DA', '#54f2f2', '#D81E5B',]
#generic_lines.reverse()
generic_lines_us.reverse()

linestyle_tuple = [
     ('solid', 'solid'),
     #('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     #('loosely dashed',        (0, (5, 10))),
     #('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     #('dashdotted',            (0, (3, 5, 1, 5))),
     #('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),]

# linestyle_tuple = [
#      ('solid', 'solid'),
#      #('loosely dotted',        (0, (1, 10))),
#      #('densely dotted',        (0, (1, 1))),
#      #('long dash with offset', (5, (10, 3))),
#      #('loosely dashed',        (0, (5, 10))),
#      #('dashed',                (0, (5, 5))),
#      ('dotted',        (0, (5, 1))),

#      #('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      #('dashdotted',            (0, (3, 5, 1, 5))),
#      #('densely dashdotted',    (0, (3, 1, 1, 1))),

#      #('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      #('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('solid', 'solid'),
#      ('dotted',                (0, (1, 1))),]

linestyles = []

for i, (name, linestyle) in enumerate(linestyle_tuple):
    linestyles.append(linestyle)

#print(len(linestyles))
#print(linestyles[0])

    

shade_color = {
    'NN_UCB': '#EB9605',#generic_lines[3],  # all_colors_genz['sharp_blue'],
    'GNN_UCB': generic_lines[0],  # all_colors_genz['sharp_green'],
    'NN_US': generic_lines[0],  # all_colors_genz['sharp_red'],
    'GNN_US': generic_lines[1]  # all_colors_genz['sharp_yellow'],
    # 'NN_UCB': all_colors_genz['soft_blue'],
    # 'GNN_UCB': all_colors_genz['soft_green'],
    # 'NN_US': all_colors_genz['soft_red'],
    # 'GNN_US': all_colors_genz['soft_yellow'],
}

line_color = {
    'NN_UCB': '#EB9605',#generic_lines[3],#all_colors_genz['sharp_blue'],
    'GNN_UCB': generic_lines[0],#all_colors_genz['sharp_green'],
    'NN_US': generic_lines[0],#all_colors_genz['sharp_red'],
    'GNN_US': generic_lines[1]#all_colors_genz['sharp_yellow'],
}


picked_var_label = {
    'GNN_UCB': r'$\hat{\sigma}(G_t) - GNN-UCB$',
    'NN_UCB': r'$\hat{\sigma}(G_t) - NN-UCB$',
    'GNN_US': r'$\hat{\sigma}(G_t) - GNN-US$',
    'NN_US': r'$\hat{\sigma}(G_t) - GNN-US$',
    'GP_UCB': r'$\hat{\sigma}(G_t) - GP-UCB$',
}
avg_var_label = {
    'GNN_UCB': r'$\bar{\hat{\sigma}}_t - GNN-UCB$',
    'NN_UCB': r'$\bar{\hat{\sigma}}_t - NN-UCB$',
    'GNN_US': r'$\bar{\hat{\sigma}}_t - GNN-US$',
    'NN_US': r'$\bar{\hat{\sigma}}_t - NN-US$',
    'GP_UCB': r'$\hat{\sigma}(G_t) - GP-UCB$',
}


linestyle = {
    'GNN_UCB': '-',
    'NN_UCB': '--',
    'GNN_US': '-.',
    'NN_US': ':'
}

algo_names = {
    'GNN_UCB': 'GNN-UCB',
    'NN_UCB': 'NN-UCB',
    'GNN_US': 'GNN-PE',
    'NN_US': 'NN-PE'
}