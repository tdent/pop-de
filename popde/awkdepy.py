import numpy as np
from density_estimate import SimpleKernelDensityEstimation
from KDEpy.TreeKDE import TreeKDE

class General_adaptiveKDE():#SimpleKernelDensityEstimation):
    """
    General adaptive Kernel Density Estimation (KDE) using KDEpy
    with variable bandwidth per point

    Example:
    # Example of 3D KDE and plot for verification
    mean1, sigma1 = 0.0, 1.0
    mean2, sigma2 = 0.0, 1.0
    mean3, sigma3 = 0.0, 1.0
    n_samples = 1000
    rndgen = np.random.RandomState(seed=1)
    sample1 = rndgen.normal(mean1, sigma1, size=n_samples)
    sample2 = rndgen.normal(mean2, sigma2, size=n_samples)
    sample3 = rndgen.normal(mean3, sigma3, size=n_samples)
    sample = np.column_stack((sample1, sample2, sample3)) # shape is (n_points, n_features)
    # Create and fit the adaptive KDE
    kde = General_adaptiveKDE(sample, dim_names=['x', 'y', 'z'], alpha=0.5, input_transf=None)
    print("kde=", kde)
    #kde.fit()
    
    # Generate grid for plotting
    xgrid = np.linspace(sample1.min(), sample1.max(), 100)
    ygrid = np.linspace(sample2.min(), sample2.max(), 100)
    zgrid = np.linspace(sample3.min(), sample3.max(), 100)
    XX, YY, ZZ = np.meshgrid(xgrid, ygrid, zgrid)
    eval_pts = np.column_stack((XX.flatten(), YY.flatten(), ZZ.flatten()))
    kde.This_fit(eval_pts)

    # Evaluate the KDE at the grid points
    density_values = kde.evaluate(eval_pts)
    density_values = density_values.reshape(XX.shape)
    print(density_values)
    """
    def __init__(self, data, backend='KDEpy', bandwidth=1., dim_names=None, alpha=0.5,  input_transf=None, stdize=False, rescale=None):
        # Additional parameter for the Wang & Wang formula
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kde_data = data
        #super().__init__(data, input_transf, stdize, rescale,
        #                 backend, bandwidth, dim_names)  

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
        print("kdevals = ", kde_values)
        g = np.exp(np.sum(np.log(kde_values)) / len(kde_values))
        inv_loc_bw = (kde_values / g) ** self.alpha
        return inv_loc_bw

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

    def This_fit(self, points):
        """
        Fit the adaptive KDE
        """
        from KDEpy.TreeKDE import TreeKDE
        #First get pilot KDE
        pilot_kde = TreeKDE(bw=self.bandwidth).fit(self.kde_data)
        pilot_values = pilot_kde.evaluate(points)
        # Calculate per-point bandwidths
        per_point_bandwidths = self.calculate_per_point_bandwidths(pilot_values)

        # Update the KDE with per-point bandwidths
        self.kernel_estimate = TreeKDE(bw=self.bandwidth).fit(self.kde_data)


    def evaluate(self, points):
        density_values = self.kernel_estimate.evaluate(points)
        return density_values

