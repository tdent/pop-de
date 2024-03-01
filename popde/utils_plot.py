import numpy as np
import matplotlib.pyplot as plt



def simple2Dplot(xx, yy, kde):
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

    return fig

