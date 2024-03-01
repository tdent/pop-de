import numpy as np
import scipy
from density_estimate import VariableBwKDEPy 
from KDEpy.TreeKDE import TreeKDE

class AdaptiveBwKDE(VariableBwKDEPy):
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
    # Create and fit the adaptive KDE note backend must be set correctly
    kde = AdaptiveBwKDE(sample, backend='awKDEpy', dim_names=['x', 'y', 'z'], alpha=0.5, input_transf=None)
    # Generate grid for plotting
    xgrid = np.linspace(sample1.min(), sample1.max(), 100)
    ygrid = np.linspace(sample2.min(), sample2.max(), 100)
    zgrid = np.linspace(sample3.min(), sample3.max(), 100)
    XX, YY, ZZ = np.meshgrid(xgrid, ygrid, zgrid)
    eval_pts = np.column_stack((XX.flatten(), YY.flatten(), ZZ.flatten()))
    kde.fit()

    # Evaluate the KDE at the grid points
    density_values = kde.evaluate(eval_pts)
    density_values = density_values.reshape(XX.shape)
    print(density_values)
    """
    def __init__(self, data, backend='KDEpy', bandwidth=1., dim_names=None, alpha=0.5,  input_transf=None, stdize=False, rescale=None):
        # Additional parameter for the Wang & Wang formula
        self.alpha = alpha
        super().__init__(data, input_transf, stdize, rescale,
                         backend, bandwidth, dim_names)  

    def _local_bandwidth_factor(self, kde_values):
        """
        Calculate local bandwidth factor using expression in
        B. Wang and X. Wang, 2007, DOI: 10.1214/154957804100000000.

        Parameters:
        -----------
        kde_values : array-like
            The KDE values at the data point positions.

        Returns:
        --------
        loc_bw_factor : array-like
           local bandwidth factor for adaptive 
           bandwidth.
        """
        from scipy.stats import gmean
        #geometric mean of kde values
        g = gmean(kde_values)
        loc_bw_factor = (kde_values / g) ** self.alpha
        return loc_bw_factor

    def calculate_per_point_bandwidths(self, pilot_values):
        """
        Calculate per-point bandwidths using the Wang & Wang formula.

        Parameters:
        -----------
        pilot_values : array-like
            The pilot KDE values at the data point positions.

        Returns:
        --------
        per_point_bandwidths : array-like
            Per-point bandwidths calculated using the Wang & Wang formula.
            bw_i =  global_bw * local_bw_factor

                local_bw_factor = (f(X_i)/g)^alpha
                f(X_i) is pilot-KDE values
                g is local factor given in local_bandwidth_factor
        """
        # Calculate the inverse of local bandwidth using the provided function
        local_bandwidths = self._local_bandwidth_factor(pilot_values)

        # Use the local bandwidths to calculate per-point bandwidths
        self.bandwidth = self.bandwidth / local_bandwidths


    def fit_awKDEpy(self):
        """
        Fit the adaptive KDE
        """
        pilot_kde = TreeKDE(bw=self.bandwidth).fit(self.kde_data)
        pilot_values = pilot_kde.evaluate(self.kde_data)
        # Calculate per-point bandwidths as  re-assigning self.bandwidth
        self.calculate_per_point_bandwidths(pilot_values)

        # Update the KDE with per-point bandwidths
        self.kernel_estimate = TreeKDE(bw=self.bandwidth).fit(self.kde_data)


    def evaluate(self, points):
        density_values = self.kernel_estimate.evaluate(points)
        return density_values


class AdaptiveKDELeaveOneOutCrossValidation():
    """
    A class that given input values of observations and a choice of
    bandwidth and adaptive parameter values, optimizes the parameters
    using leave-one-out cross validation and evaluates the density
    estimate

    Methods:
       kde_awkde(evaluate_grid): train and fit a kde return kde object
       loocv(bw, alpha): find best bw from grid of values using
           cross validated likelihood
       estimation(): optimize and return kde values on grid of values

    Data attributes:
        data: input data from observation [_ndimensions]
        bw : bandwidth choices, sequence
        alpha: adaptive parameter, sequence
        kwarg: flag for log parameter
    Usage:
    >>> xvals = np.random.rand(5,5)
    >>> bwarr = np.arange(0, 1.1, 0.1)
    >>> densityestimator.__init__(xvals, bwarray)
    """
    def __init__(self, data, bw_arr, alpha_arr, paramflag=None):  # FIXME allow for log/non-log in different dimensions
        self.data = data
        self.bw_arr = bw_arr
        self.alpha_arr = alpha_arr
        self.log_param = paramflag
        self.optbw = None
        self.optalpha = None

    def train_eval_kde(self, x, x_grid, bandwidth, alpha, ret_kde=False):
        """Kernel Density Estimation with awkde
        inputs:
        x: training data [n_dimension]
        x_eval: evaluation points
        bandwidth: global bw for kde
        alpha: smoothing parameter
        kwargs:
        ret_kde
            if True kde will be output with estimated kde-values
        """
        from awkde import GaussianKDE
        kde = GaussianKDE(glob_bw=gl_bandwidth)
        kde.fit(x[:, np.newaxis])
        # Evaluate KDE at given points
        if isinstance(x_eval, (list, tuple, np.ndarray)) == False:
            y = kde.predict(x_eval)
        else:
            y = kde.predict(x_eval[:, np.newaxis])

        if ret_kde == True:
            return kde, y
        return y

    def loocv(self, sample, bw, alpha):
        """
        Calculate likelihood FOM using leave one out cross validation for
        finding best choice of bandwidth and alpha
        """
        fom = 0.
        for i in range(len(sample)):
            leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
            y = train_eval_kde(leave_one_sample, miss_sample, bw, alpha)
            fom += np.log(y)
        return fom

    def estimation(self, samples, x_eval, alphagrid, bwgrid):
        """
        given input data compute kde with optimization
        over the bandwidth and alpha using cross validation
        return kde evaluated on desired domain
        if paramflag are log, train KDE on log of input values
        """
        if self.log_param == 'log':  # FIXME: allow different dimensions with/without log
            samples = np.log(samples)
        fom = {}  # Dictionary holding optimization figure of merit
        for bw in bwgrid:
            for alp in alphagrid:
                fom[(gbw, alp)] = loocv(samples, bw, alp)
        optvalues = max(fom.items(), key=operator.itemgetter(1))[0]
        self.optbw, self.optalpha = optvalues[0], optvalues[1]
        kdeval = train_eval_kde(samples, x_eval, self.optbw, self.optalpha)

        return kdeval, self.optbw, self.optalpha
