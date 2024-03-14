def _from_base(*, base):
    return {'axes.axisbelow': True,
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.5,
            'legend.edgecolor': 'inherit',
            'lines.linewidth': 3.0,
            'patch.linewidth': 0.5,
            'xtick.major.size': 3.0,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 2.0,
            'xtick.minor.width': 0.25,
            'ytick.major.size': 3.0,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 2.0,
            'ytick.minor.width': 0.25}
def icml2022():
    return  _from_base(base=10)