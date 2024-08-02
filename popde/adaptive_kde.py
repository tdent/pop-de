import numpy as np
import scipy
from density_estimate import VariableBwKDEPy 
from scipy.stats import gmean


class AdaptiveBwKDE(VariableBwKDEPy):
    """
    General adaptive Kernel Density Estimation (KDE) using KDEpy
    with variable bandwidth per point
    The variation is controlled by an additional parameter 'alpha', see B. Wang and X. Wang, 2007, DOI: 10.1214/154957804100000000.

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
    kde = AdaptiveBwKDE(sample, dim_names=['x', 'y', 'z'], alpha=0.5, input_transf=None)
    # Generate grid for plotting
    xgrid = np.linspace(sample1.min(), sample1.max(), 100)
    ygrid = np.linspace(sample2.min(), sample2.max(), 100)
    zgrid = np.linspace(sample3.min(), sample3.max(), 100)
    XX, YY, ZZ = np.meshgrid(xgrid, ygrid, zgrid)
    eval_pts = np.column_stack((XX.flatten(), YY.flatten(), ZZ.flatten()))

    # Evaluate the KDE at the grid points
    density_values = kde.evaluate(eval_pts)
    density_values = density_values.reshape(XX.shape)
    print(density_values)
    """
    def __init__(self, data, weights, input_transf=None, stdize=False,
                 rescale=None, backend='KDEpy', bandwidth=1., alpha=0.5, dim_names=None,
                 do_fit=True):
        self.alpha = alpha
        self.global_bandwidth = bandwidth
        self.pilot_values = None

        # Set up pilot KDE with fixed bandwidth
        # If do_fit is True, fit the pilot KDE; if not, just initialize
        self.pilot_kde = VariableBwKDEPy(data, weights, input_transf, stdize, rescale,
                                         backend, bandwidth, dim_names, do_fit)
        # Initialize the adaptive KDE
        super().__init__(data, weights, input_transf, stdize, rescale, backend,
                         bandwidth, dim_names, do_fit=False)
        # Special initial fit method
        # Note that self.fit() is inherited from the parent & uses self.bandwidth directly
        if do_fit:
            self.fit_adaptive()
    
    def fit_adaptive(self):
        # Make sure pilot KDE has been fit
        if self.pilot_kde.kernel_estimate is None:
            self.pilot_kde.fit()
        # Compute pilot kde values at input points
        self.pilot_values = self.pilot_kde.evaluate(self.kde_data)
        # Calculate per-point bandwidths and apply them to fit adaptive KDE
        self.set_per_point_bandwidth(self.pilot_values)

    def _local_bandwidth_factor(self, kde_values):
        """
        Calculate local bandwidth factor using expression in
        B. Wang and X. Wang, 2007, DOI: 10.1214/154957804100000000.

        Parameters
        ----------
        pilot KDE_values : array-like
            The KDE values at the data point positions.

        Returns
        -------
        loc_bw_factor : array-like
           local bandwidth factor for adaptive 
           bandwidth.
        """
        # Geometric mean of pilot kde values 
        g = gmean(kde_values)
        loc_bw_factor = (kde_values / g) ** self.alpha
        return loc_bw_factor

    def set_per_point_bandwidth(self, pilot_values):
        """
        Calculate per-point bandwidths and re-fit KDE

        Parameters
        ----------
        pilot_values : array-like
            The pilot KDE values at the data point positions.
        """
        self.set_bandwidth(
            self.global_bandwidth / self._local_bandwidth_factor(pilot_values)
        )
    
    def set_alpha(self, new_alpha):
        """
        Set the adaptive parameter alpha to a new value and re-initialize KDE.

        Parameters
        ----------
        new_alpha : float
            The new value for the adaptive parameter alpha.
        """
        self.alpha = new_alpha
        # Make sure pilot KDE was evaluated
        if self.pilot_values is None:
            self.pilot_kde.set_bandwidth(self.global_bandwidth)
            self.pilot_values = self.pilot_kde.evaluate(self.kde_data)
        # Update local bandwidths
        self.set_per_point_bandwidth(self.pilot_values)       

    def set_adaptive_parameter(self, new_alpha, new_global_bw):
        """
        Update the adaptive parameter alpha and the global bandwidth,
        then reassign local bandwidths and re-initialize the KDE.

        Parameters
        ----------
        new_alpha : float
            The new value for the adaptive parameter alpha.
        new_global_bw : float
            The new value for the global bandwidth.
        """
        self.global_bandwidth = new_global_bw
        
        # Re-fit pilot KDE
        self.pilot_kde.set_bandwidth(new_global_bw)
        # Update pilot_values
        self.pilot_values = self.pilot_kde.evaluate(self.kde_data)
        # Set alpha and calculate per-point bandwidths 
        self.set_alpha(new_alpha)


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

