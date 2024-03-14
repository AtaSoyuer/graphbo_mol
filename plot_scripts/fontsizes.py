
def _from_base(*, base):
    return {
        "font.size": base - 1,
        "axes.labelsize": base - 1,
        "legend.fontsize": base - 3,
        "xtick.labelsize": base - 5,
        "ytick.labelsize": base - 5,
        "axes.titlesize": base - 2,
    }
def neurips2022():
    return  _from_base(base=13)

def _from_base_icml(*, base):
    return {
        "font.size": base - 1, #9
        "axes.labelsize": base - 3, #9
        "legend.fontsize": base - 1, # 7
        "xtick.labelsize": base - 2, # 7
        "ytick.labelsize": base - 2, # 7
        "axes.titlesize": base - 1, # 9
    }
def icml2022():
    return  _from_base_icml(base=11)
