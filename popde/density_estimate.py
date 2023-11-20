class AdaptiveKDEWithBandWidthUsingLeaveOneOutCrossValidation():
    """
    A class that given input values of observations and choices of
    bw for each dimension of input data
    compute density estimator with bandwidth optimized 
    using cross validation

    Methods:
       kde_awkde(evaluate_grid): train and fit a kde return kde object
       loocv(bw, alpha): find best bw from grid of values
       estimation(): return kde values on grid of values

    Data attributes:
        data: input data from observation [_ndimensions]
        bw : bw choices [_ndimensions]
        alpha: smoothing parameter for adaptive bandwidth
        minX: min of values on which to evaluate kde
        maxX: max of values on which to evaluate kde
        kwarg: flag on parameter units
    Usage:
    >>> xvals = np.random.rand(5,5)
    >>> bwarr = np.arange(0, 1.1, 0.1)
    >>> densityestimator.__init__(xvals, bwarray)
    """
    def __init__(self, data, bw, alpha, minX, maxX, paramflag='log'):
        self.data = data
        self.bw = bw
        slef.alpha = alpha
        self.minX = 0 #in multi-Dimensions we need to have list with each dimension min val
        self.maxX =100

    def kde_awkde(self, x, x_grid, bandwidth, alpha, ret_kde=False):
        """Kernel Density Estimation with awkde 
        inputs:
        x = training data [n_dimension] 
        x_grid = testing data
        bandwidth = global bw for kde
        alpha =smoothing parameter
        kwargs:
        ret_kde optional 
            if True kde will be output with estimated kde-values 
        """
        from awkde import GaussianKDE  
        kde = GaussianKDE(glob_bw=gl_bandwidth)
        kde.fit(x[:, np.newaxis])
        if isinstance(x_grid, (list, tuple, np.ndarray)) == False:
            y = kde.predict(x_grid)
        else:
            y = kde.predict(x_grid[:, np.newaxis])

        if ret_kde == True:
            return kde, y
        return y

    def loocv_awkde(self, sample, bwchoice, alphachoice):
        """
        use of leave one out cross validation for 
        finding best choice of bandwidth and alpha
        """
        fom = 0.0
        for i in range(len(sample)):
            leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
            y = kde_awkde(leave_one_sample, miss_sample, alp=alphachoice, gl_bandwidth=bwchoice)
            fom += np.log(y)
        return fom
    
    def estimation(self, samplevalues, x_gridvalues, alphagrid, bwgrid, paramflag):
        """
        given input data compute kde with optimization
        over the bandwidth and alpha using cross validation
        return kde evaluated on desired domain
        if paramflag are log use log of input values
        to get KDE
        """
        if paramflag=='log':
            samples = np.log(samplevalues)
        figure_of_merit = {}
        for gbw in bwgrid:
            for alp in alphagrid:
                FOM[(gbw, alp)] = loocv_awkde(sample, gbw, alp)
        optvalues = max(FOM.items(), key=operator.itemgetter(1))[0]
        optbw, optalpha  = optvalues[0], optvalues[1]

        kdeval = kde_awkde(samples, x_gridvalues, alp=optalpha, gl_bandwidth=optbw)
    return kdeval, optbw, optalpha

import numpy as np
from scipy.stats import norm

class SimpleGaussianKernelDensityEstimation:
    """
    Kernel Density Estimation (KDE) using Gaussian Kernels

    Parameters:
    -----------
    data : array-like
        The input data for which KDE will be estimated.

    bandwidth : float, optional (default=1.0)
        Bandwidth parameter controlling the width of the kernels.
        Larger bandwidths lead to smoother density estimates.

    kernel : str, optional (default='gaussian')
        The kernel function used for smoothing. Supported kernels: 'gaussian',

    Attributes:
    -----------
    data : array-like
        The input data for which KDE is estimated.

    bandwidth : float
        Bandwidth parameter controlling the width of the kernels.

    kernel : str
        The kernel function used for smoothing.

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
    kde = KernelDensityEstimation(data)
    kde.fit()

    # Evaluate the KDE at new data points
    new_data = np.linspace(-3, 3, 100)
    density_values = kde.evaluate(new_data)
    """

    def __init__(self, data, bandwidth=1.0, kernel='gaussian'):
        """
        Initialize the KernelDensityEstimation object.

        Parameters:
        -----------
        data : array-like
            The input data for which KDE will be estimated.

        bandwidth : float, optional (default=1.0)
            Bandwidth parameter controlling the width of the kernels.
            Larger bandwidths lead to smoother density estimates.

        kernel : str, optional (default='gaussian')
            The kernel function used for smoothing.
        """
        
        import numpy as np
        from scipy.stats import norm
        self.data = np.asarray(data)
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, data=None, bandwidth=None, kernel=None):
        """
        Fit the KDE to new data.

        Parameters:
        -----------
        data : array-like, optional
            The input data for which KDE will be estimated. If None, the data provided during initialization is used.

        bandwidth : float, optional
            Bandwidth parameter controlling the width of the kernels.
            Larger bandwidths lead to smoother density estimates. If None, the bandwidth provided during initialization is used.

        kernel : str, optional
            The kernel function used for smoothing. If None, the kernel provided during initialization is used.
        """
        if data is not None:
            self.data = np.asarray(data)
        if bandwidth is not None:
            self.bandwidth = bandwidth
        if kernel is not None:
            self.kernel = kernel

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
        points = np.asarray(points)
        if self.kernel == 'gaussian':
            kernel_function = norm(loc=0, scale=self.bandwidth).pdf
        else:
            raise ValueError("Unsupported kernel. Supported kernels: 'gaussian',")

        density_values = np.zeros_like(points, dtype=float)

        for data_point in self.data:
            density_values += kernel_function((points - data_point) / self.bandwidth)

        # Normalize by the number of data points and bandwidth
        density_values /= (len(self.data) * self.bandwidth)

        return density_values

# Example usage:
# data = np.random.randn(100)
# kde = SimpleGaussianKernelDensityEstimation(data)
# kde.fit()
# new_data = np.linspace(-3, 3, 100)
# density_values = kde.evaluate(new_data)

