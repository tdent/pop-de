import numpy as np
from scipy.stats import gaussian_kde
from density_estimate import SimpleKernelDensityEstimation

class AdaptiveKernelDensityEstimation(SimpleKernelDensityEstimation):
    """
    Subclass of SimpleKernelDensityEstimation to implement a general adaptive KDE
    with per-point bandwidths using the Wang & Wang formula https://github.com/mennthor/awkde/tree/master).

    Methods:
    --------
    evaluate(points):
        Evaluate the adaptive KDE at given data points.

    fit():
        Fit the adaptive KDE using the Wang & Wang formula for per-point bandwidths.

    _calc_local_bandwidth():
        Calculate local bandwidth using the provided function.

    Attributes:
    -----------
    pilot_kde : scipy.stats.gaussian_kde
        The initial KDE used as a pilot with fixed scalar bandwidth.
    """

    def __init__(self, data, input_transf=None, stdize=False, rescale=None,
                 backend='scipy', bandwidth=1., dim_names=None, alpha=0.5):
        """
        Initialize the AdaptiveKernelDensityEstimation class.

        Parameters:
        -----------
        Same as SimpleKernelDensityEstimation with addition of:
        pilot_bandwidth : float
            Bandwidth used for the initial pilot KDE.
        alpha : float
            Adaptive parameter for locally adaptive bandwidth calculation.
        """
        super().__init__(data, input_transf, stdize, rescale, backend, bandwidth, dim_names)
        self.alpha = alpha

        # Additional attribute for the bandwidth of the initial pilot KDE
        self.pilot_bandwidth = 1.0  #set this as  initial pilot bandwidth 

        # Initialize the pilot KDE
        self.fit_pilot()

    def _calc_local_bandwidth(self, kde_values):
        """
        Calculate local bandwidth using the provided function.

        Parameters:
        -----------
        kde_values : array-like
            The KDE values at the data point positions.

        Returns:
        --------
        inv_loc_bw : array-like
            Inverse of the local bandwidth calculated using the provided function.
        """
        g = np.exp(np.sum(np.log(kde_values)) / len(kde_values))
        inv_loc_bw = (kde_values / g) ** self.alpha
        return inv_loc_bw

    def fit_pilot(self):
        """
        Fit the initial pilot KDE with a fixed scalar bandwidth.
        """
        self.pilot_kde = gaussian_kde(self.kde_data.T, bw_method=self.pilot_bandwidth)

    def fit(self, adaptive=True):
        """
        Fit the adaptive KDE using the Wang & Wang formula for per-point bandwidths.

        Parameters:
        -----------
        adaptive : bool, optional (default=True)
            If True, use adaptive bandwidth; if False, use fixed bandwidth.
        """
        # Calculate the pilot KDE values at the data point positions
        pilot_values = self.pilot_kde(self.kde_data.T)

        # Use Wang & Wang formula to calculate per-point bandwidths
        per_point_bandwidths = self.calculate_per_point_bandwidths(pilot_values, adaptive=adaptive)

        # Fit the adaptive KDE with per-point bandwidths
        self.kernel_estimate = gaussian_kde(self.kde_data.T, bw_method=per_point_bandwidths)

    def calculate_per_point_bandwidths(self, pilot_values, adaptive=True):
        """
        Calculate per-point bandwidths using the Wang & Wang formula.

        Parameters:
        -----------
        pilot_values : array-like
            The pilot KDE values at the data point positions.
        adaptive : bool, optional (default=True)
            If True, use adaptive bandwidth; if False, use fixed bandwidth.

        Returns:
        --------
        per_point_bandwidths : array-like
            Per-point bandwidths calculated using the Wang & Wang formula.
        """
        # Calculate the local bandwidth using the provided function
        local_bandwidths = self._calc_local_bandwidth(pilot_values)

        # Use the local bandwidths to calculate per-point bandwidths
        if adaptive:
            per_point_bandwidths = self.bandwidth * local_bandwidths
        else:
            per_point_bandwidths = np.ones_like(local_bandwidths) / self.bandwidth

        return per_point_bandwidths

# Example usage:
np.random.seed(42)
sample_data_3d = np.random.multivariate_normal([0.5, 1.0, 0.5], np.eye(3), size=1000)
adaptive_kde = AdaptiveKernelDensityEstimation(sample_data_3d, dim_names=['x', 'y', 'z'], alpha=0.5)

# Fit the KDE with adaptive bandwidth
adaptive_kde.fit(adaptive=True)


#### An example to test
from mpl_toolkits.mplot3d import Axes3D
# Generate random 3D data
mean = [0, 0, 0]
covariance = [[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]
np.random.seed(42)
sample_data_3d = np.random.multivariate_normal(mean, covariance, size=1000)

# Create an instance of AdaptiveKernelDensityEstimation
adaptive_kde = AdaptiveKernelDensityEstimation(sample_data_3d, dim_names=['x', 'y', 'z'], alpha=0.5)

# Evaluate the KDE on a grid
grid_size = 50
x_grid = np.linspace(-3, 3, grid_size)
y_grid = np.linspace(-3, 3, grid_size)
z_grid = np.linspace(-3, 3, grid_size)
grid_points = np.array(np.meshgrid(x_grid, y_grid, z_grid)).T.reshape(-1, 3)

# Evaluate the adaptive KDE at the grid points
density_values = adaptive_kde.evaluate(grid_points)
# Reshape the density values for plotting
density_grid = density_values.reshape((grid_size, grid_size, grid_size))
# Plot the 3D KDE
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample_data_3d[:, 0], sample_data_3d[:, 1], sample_data_3d[:, 2], s=5, color='blue', label='Data points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D KDE with Adaptive Bandwidths')

# Plot the density values using a 3D surface plot
ax.plot_surface(x_grid, y_grid, z_grid, density_grid, cmap='viridis', alpha=0.7)
plt.show()
