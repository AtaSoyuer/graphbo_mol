import figsizes, fonts, fontsizes,linestyles
import matplotlib.font_manager


def neurips2022(*,  rel_width=1.0, nrows=1, ncols=1, family="DejaVu Sans", tight_layout = False):
    """Neurips 2022 bundle."""
    font_config = fonts.neurips2022(family=family)
    size = figsizes.neurips2022(rel_width=rel_width, nrows=nrows, ncols=ncols, tight_layout=tight_layout)
    fontsize_config = fontsizes.neurips2022()
    return {**font_config, **size, **fontsize_config}

def icml2022(*,  rel_width=1.0, nrows=1, ncols=1, family="DejaVu Sans", tight_layout = False):
    """Neurips 2022 bundle."""
    font_config = fonts.icml2022(family=family)
    size = figsizes.icml2022(rel_width=rel_width, nrows=nrows, ncols=ncols, tight_layout=tight_layout)
    fontsize_config = fontsizes.icml2022()
    line_config = linestyles.icml2022()
    return {**font_config, **size, **fontsize_config, **line_config}