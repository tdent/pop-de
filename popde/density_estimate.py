import numpy as np

class SimpleKernelDensityEstimation:
    """
    Fit and evaluate multi-dimensional Kernel Density Estimation (KDE) 

    Methods:
    --------
    check_dimensionality():
        check if the data matches the dimensions of KDE.

    evaluate(points):
        Evaluate the KDE at given data points.

    Examples:
    ---------
    # Create a KDE instance and fit it to data
    data = np.random.randn((100, 100))
    kde = SimpleGaussianKernelDensityEstimation(data)

    # Evaluate the KDE at new data points
    new_data = np.array(np.linspace(-3, 3, 100).tolist(), np.linspace(-3, 3, 100).tolist())
    density_values = kde.evaluate(new_data)
    """
    def __init__(self, data, bandwidth=1.0, kernel='gaussian', dim_names=None):
        """
        Initialize the KernelDensityEstimation object.
        data: array-like, shape (n_samples, n_features)
               points of the data define each kernel position
               each row is a point, eachcolumn is a feature.
        kwargs:
            dim_names : sequence of dimension names, e.g. ('m1', 'z', 'chi_eff') 
                        values must be strings
        """
        if len(data.shape) != 2:
            raise ValueError("data must have shape (n_samples, n_features).")

        self.data = np.asarray(data) 
        self.bandwidth = bandwidth
        self.kernel = kernel 

        self.dim_names = dim_names
        if dim_names is not None:
            self.check_dimensionality()

   def check_dimensionality(self):
       """
       Check if the dimension of KDE matches the parameters on which we evaluate KDE.
        Raises ValueError
       """
        if len(self.data.shape[0]) != len(self.dim_names):
            raise ValueError("Dimensionality of data array does not match the number of dimension names.")

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
            using standard gaussian distribution
        """
        import scipy
        from scipy.stats import gaussian_kde 

        # scipy takes data with shape (n_dimensions, n_samples)
        kernel_function = gaussian_kde(self.data.T, bw_method=self.bandwidth)
        points = np.asarray(points).T
        density_values = kernel_function(points)

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
