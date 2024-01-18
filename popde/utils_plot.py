import numpy as np
import matplotlib.pyplot as plt



def simple2Dplot(xx, yy, kde):
    """
    return kde plot in 2D given output
    """
    fig = plt.figure()
    ax = fig.gca()
    # Contourf plot
    cfset = ax.contourf(xx, yy, kdevals, cmap='Blues')
    # Contour plot
    cset = ax.contour(xx, yy, kdevals, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('Y1')
    ax.set_ylabel('Y0')

    plt.show()

