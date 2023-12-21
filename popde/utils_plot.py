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


class KDEPlotter:
    def __init__(self, kde):
        """
        Initialize the KDE plotter with a given KDE instance.

        Parameters:
        -----------
        kde : SimpleKernelDensityEstimation
            The KDE instance to be plotted.
        """
        self.kde = kde

    def plot_1d_projection(self, dim_idx, num_points=100, **kwargs):
        """
        Plot the 1D projection of the KDE along a specified dimension.

        Parameters:
        -----------
        dim_idx : int
            The index of the dimension along which to project.
        num_points : int, optional
            The number of points to evaluate the projection, default is 100.
        **kwargs:
            Additional keyword arguments to pass to the plot function.
        """
        values = np.linspace(np.min(self.kde.data[:, dim_idx]),
                             np.max(self.kde.data[:, dim_idx]),
                             num_points)
        points = np.zeros((num_points, self.kde.data.shape[1]))
        points[:, dim_idx] = values
        density_values = self.kde.evaluate(points)

        plt.plot(values, density_values, **kwargs)
        plt.xlabel(self.kde.dim_names[dim_idx] if self.kde.dim_names else f'Dimension {dim_idx + 1}')
        plt.ylabel('Density')
        plt.title(f'1D Projection along {self.kde.dim_names[dim_idx]}'
                  if self.kde.dim_names else f'1D Projection along Dimension {dim_idx + 1}')
        plt.show()

    def plot_2d_slice(self, dim1_idx, dim2_idx, num_points=100, **kwargs):
        """
        Plot a 2D slice of the KDE by fixing two dimensions.

        Parameters:
        -----------
        dim1_idx : int
            The index of the first dimension.
        dim2_idx : int
            The index of the second dimension.
        num_points : int, optional
            The number of points to evaluate the slice, default is 100.
        **kwargs:
            Additional keyword arguments to pass to the plot function.
        """
        values_dim1 = np.linspace(np.min(self.kde.data[:, dim1_idx]),
                                  np.max(self.kde.data[:, dim1_idx]),
                                  num_points)
        values_dim2 = np.linspace(np.min(self.kde.data[:, dim2_idx]),
                                  np.max(self.kde.data[:, dim2_idx]),
                                  num_points)
        points = np.zeros((num_points ** 2, self.kde.data.shape[1]))
        points[:, dim1_idx] = np.tile(values_dim1, num_points)
        points[:, dim2_idx] = np.repeat(values_dim2, num_points)
        density_values = self.kde.evaluate(points)

        plt.tricontourf(points[:, dim1_idx], points[:, dim2_idx], density_values, **kwargs)
        plt.xlabel(self.kde.dim_names[dim1_idx] if self.kde.dim_names else f'Dimension {dim1_idx + 1}')
        plt.ylabel(self.kde.dim_names[dim2_idx] if self.kde.dim_names else f'Dimension {dim2_idx + 1}')
        plt.title(f'2D Slice along {self.kde.dim_names[dim1_idx]} and {self.kde.dim_names[dim2_idx]}'
                  if self.kde.dim_names else f'2D Slice along Dimensions {dim1_idx + 1} and {dim2_idx + 1}')
        plt.show()

# Example usage:
from density_estimate  import  SimpleKernelDensityEstimation
mean1, sigma1 = 14.0, 1.5
mean2, sigma2 = 3.0, 0.25
n_samples = 1000
rndgen = np.random.RandomState(seed=1)
sample1 = rndgen.normal(mean1, sigma1, size=n_samples)
sample2 = rndgen.normal(mean2, sigma2, size=n_samples)
data= np.column_stack((sample1, sample2)) # shape is (n_points, n_features)
kde_instance = SimpleKernelDensityEstimation(data, dim_names=['m1', 'z'])
kde_plotter = KDEPlotter(kde_instance)
kde_plotter.plot_1d_projection(0)
kde_plotter.plot_2d_slice(0, 1)


class DataVisualizer:
    def __init__(self, density_estimation, data_processor):
        self.density_estimation = density_estimation
        self.data_processor = data_processor

    def plot_density(self, ax=None, **kwargs):
        if ax is None:
            ax = self.get_default_axes()

        self.plot_density_on_axes(ax, **kwargs)
        self.label_axes(ax, xlabel='X-axis', ylabel='Y-axis', title='Density Plot')
        self.show_plot()

    def plot_slices(self, constant_values, ax=None, **kwargs):
        if ax is None:
            ax = self.get_default_axes()

        for value in constant_values:
            slice_data = self.get_slice(value)
            self.plot_points(slice_data, ax, label=f'Slice at {value}', **kwargs)

        self.label_axes(ax, xlabel='X-axis', ylabel='Y-axis', title='Slices Plot')
        self.show_plot()

    def integrate_kde(self, variables):
        print("in progress")
        # Code to integrate/marginalize KDE over one or more variables
        # ...

    def get_default_axes(self):
        print("in progress")
        # Code to initialize default plotting settings
        # ...

    def plot_density_on_axes(self, ax, **kwargs):
        print("in progress")
        # Code to plot density estimation on the given axes
        # ...

    def plot_points(self, data, ax, **kwargs):
        print("in progress")
        # Code to plot data points on the given axes
        # ...

    def label_axes(self, ax, xlabel='', ylabel='', title=''):
        print("in progress")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def show_plot(self):
        plt.show()


