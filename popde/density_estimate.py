class SimpleKernelDensityEstimation:
    """
    Fit and evaluate general Kernel Density Estimation (KDE) 

    Methods:
    --------
    fit(data, bandwidth=None, kernel=None):
        Fit the KDE to new data.

    evaluate(points):
        Evaluate the KDE at given data points.

    Examples:
    ---------
    # Create a KDE instance and fit it to data
    data = np.random.randn(100)
    kde = SimpleGaussianKernelDensityEstimation(data)
    kde.fit()

    # Evaluate the KDE at new data points
    new_data = np.linspace(-3, 3, 100)
    density_values = kde.evaluate(new_data)
    """

    def __init__(self, data, bandwidth=1.0, kernel=['gaussian'], **kwargs):
        """
        Initialize the KernelDensityEstimation object.

        Parameters:
        -----------
        data : array-like
            The input data for which KDE will be estimated.

        bandwidth : float, optional (default=1.0)
            Bandwidth parameter controlling the width of the kernels.
            Larger bandwidths lead to smoother density estimates.

        kernel : list of str, optional (default=['gaussian'])
            The kernel function used for smoothing.
        """
        import numpy as np

        self.data = np.asarray(data)
        self.bandwidth = bandwidth
        if self.kernel in kernel:
            self.kernel = kernel[0]#may need to fix this
        else:
            raise NotImplementedError("Unsupported kernel. Supported kernels: 'gaussian',")
        

    def evaluate(self, points):
        """
        Evaluate the KDE at given data points.

        Parameters:
        -----------
        points : array-like
            The data points at which the KDE will be evaluated.

        Returns:
        --------
        density_values : array-like
            The estimated density values at the given data points.
        """
        import scipy
        from scipy.stats import norm

        points = np.asarray(points)
        kernel_function = norm(loc=0, scale=self.bandwidth).pdf

        density_values = np.zeros_like(points, dtype=float)

        for data_point in self.data:
            density_values += kernel_function((points - data_point) / self.bandwidth)

        # Normalize by the number of data points and bandwidth
        density_values /= (len(self.data) * self.bandwidth)

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
