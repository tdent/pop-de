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


def get_twoD_grid(data_dim1, data_dim2, num_points=100):
    """
    return mesh grid of data 
    in two dimensions x, y 
    return  xx, yy
    """
    dim1_min, dim1_max = np.min(data_dim1), np.max(data_dim1)
    dim2_min, dim2_max = np.min(data_dim2), np.max(data_dim2)
    dim1_grid = np.linspace(dim1_min, dim1_max, num_points)
    dim2_grid = np.linspace(dim2_min, dim2_max, num_points)
    xx, yy = np.meshgrid(dim1_grid, dim2_grid)
    return xx, yy

