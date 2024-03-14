import numpy as np
import matplotlib.pyplot as plt


def simple2Dplot(xx, yy, kdevals, xlabel=None, ylabel=None, title=None, show_plot=False):
    """
    return kde plot in 2D given output
    """
    fig = plt.figure()
    ax = fig.gca()
    # Filled Contour plot
    cfset = ax.contourf(xx, yy, kdevals, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, kdevals, colors='k')
    # Contour line labels
    ax.clabel(cset, inline=1, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_plot==True:
        plt.show()
    return fig


def get_twoD_grid(data_dim1, data_dim2, num_points=100, pad=None, dim1_range=None, dim2_range=None):
    """
    return mesh grid of data 
    in two dimensions x, y 
    return  xx, yy
    """
    if dim1_range is not None:
        dim1_min, dim1_max = dim1_range
    else:
        dim1_min, dim1_max = np.min(data_dim1), np.max(data_dim1)

    if dim2_range is not None:
        dim2_min, dim2_max = dim2_range
    else:
        dim2_min, dim2_max = np.min(data_dim2), np.max(data_dim2)

    if pad is not None:
        dim1_range = dim1_max - dim1_min
        dim2_range = dim2_max - dim2_min

        dim1_min -= pad * dim1_range
        dim1_max += pad * dim1_range

        dim2_min -= pad * dim2_range
        dim2_max += pad * dim2_range

    dim1_grid = np.linspace(dim1_min, dim1_max, num_points)
    dim2_grid = np.linspace(dim2_min, dim2_max, num_points)
    xx, yy = np.meshgrid(dim1_grid, dim2_grid)
    return xx, yy

