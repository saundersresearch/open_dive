import numpy as np
import matplotlib as mpl
from matplotlib import colors
from importlib.resources import files


def generate_slant_cmap(slant_path):
    """Return SLANT colormap"""
    # Add invalid color as black
    colortable = np.loadtxt(slant_path, delimiter="|")
    colortable = colortable / 256.0  # Normalize to [0, 1]

    # Generate colormap
    cmap = colors.ListedColormap(colortable, name="slant")

    return cmap


slant_path = files("open_dive").joinpath("slant.txt")
slant_cmap = generate_slant_cmap(slant_path)
mpl.colormaps.register(name="slant", cmap=slant_cmap)
