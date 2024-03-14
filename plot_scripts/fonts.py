from matplotlib import font_manager

def _neurips_common(*, family="serif"):
    """Default fonts for Neurips."""
    return {
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # free ptmx replacement, for ICML and NeurIPS
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }

def neurips2022(*, family="serif"):
    """Fonts for Neurips 2022."""
    return _neurips_common(family=family)


def _icml_common(*, family="serif"):
    """Default fonts for Neurips."""

    return {'font.family': 'DejaVu Serif',
        'font.serif': ['Times'],
        'mathtext.bf': 'Times:bold',
        'mathtext.fontset': 'stix',
        'mathtext.it': 'Times:italic',
        'mathtext.rm': 'Times',
        'text.usetex': False}
    

def icml2022(*, family="serif"):
    """Fonts for Neurips 2022."""
    return _icml_common(family=family)
