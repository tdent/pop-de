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
    """
    def __init__(self, data, backend='scipy', bandwidth=1., dim_names=None):
        """
        data: array-like, shape (n_samples, n_features)
               points of the data define each kernel position
               each row is a point, each column is a parameter.
        kwargs:
           bandwidth : The bandwidth of the kernel used for smoothing
            dim_names : sequence of dimension names, e.g. ('m1', 'z', 'chi_eff')
                        values must be strings
        Example:
        --------

          #For two dimensional case
          rndgen = np.random.RandomState(seed=3575)
          mean1, sigma1 = 3.0, .25
          mean2, sigma2 = 14.0, 1.5
          n_samples = 1000
          sample1 = rndgen.normal(mean1, sigma1, size=n_samples)
          sample2 = rndgen.normal(mean2, sigma2, size=n_samples)
          sample = np.column_stack((sample1, sample2)) #shape of data (n_points, n_features)
          kde = SimpleKernelDensityEstimation(sample, dim_names=['mass1', 'mass2'])
          minx, maxx = np.amin(sample1), np.amax(sample1)
          miny, maxy = np.amin(sample2), np.amax(sample2)
          x = np.linspace(minx, maxx, 100)
          y = np.linspace(miny, maxy, 100)
          XX, YY = np.meshgrid(x, y)
          eval_pts = np.column_stack((XX.flatten(), YY.flatten()))
          zz = kde.evaluate(grid_pts)
          ZZ = zz.reshape(XX.shape)
          import matplotlib.pyplot as plt
          from matplotlib.colors import LogNorm
          c = plt.pcolormesh(XX, YY, ZZ, cmap="Blues", norm=LogNorm(), shading='flat')
          plt.colorbar(c)
          plt.scatter(sample1, sample2, s=2, marker='+', c='white')
          plt.xlabel('m1')
          plt.ylabel('m2')
          plt.show()
        """
        if len(data.shape) != 2:
            raise ValueError("Data must have shape (n_samples, n_features).")

        self.data = np.asarray(data)
        self.backend = backend
        self.bandwidth = bandwidth
        self.dim_names = dim_names
        if dim_names is not None:
            self.check_dimensionality()

        self.kernel_estimate = None
        # Initialize the KDE
        self.fit()

    def check_dimensionality(self):
        """
        Check if the dimension of training data matches the number of param names
        """
        if self.data.shape[1] != len(self.dim_names):
            raise ValueError("Dimensionality of data array does not match "
                             "the number of dimension names.")

    def fit(self):
        """
        General fit method allowing for different backends
        """
        fit_method = getattr(self, 'fit_' + self.backend)
        fit_method()

    def fit_scipy(self):
        from scipy.stats import gaussian_kde

        # scipy takes data with shape (n_dimensions, n_samples)
        self.kernel_estimate = gaussian_kde(self.data.T, bw_method=self.bandwidth)

    def set_bandwidth(self, bandwidth):
        """
        Change bandwidth and re-initialize the KDE
        """
        self.bandwidth = bandwidth
        self.fit()

    def evaluate(self, points):
        """
        Evaluate the KDE allowing for different backends

        Parameters:
        -----------
        points : array-like
            The parameter points at which the KDE will be evaluated

        Returns:
        --------
        density_values : array-like
            The estimated density values at the given points
            using a standard gaussian kernel
        """
        evaluate_method = getattr(self, 'evaluate_' + self.backend)
        return evaluate_method(points)

    def evaluate_scipy(self, points):
        """
        Evaluate using scipy gaussian KDE
        """
        # scipy takes data with shape (n_dimensions, n_samples)
        points = np.asarray(points).T
        density_values = self.kernel_estimate(points)

        return density_values


class VariableBwKDEPy(SimpleKernelDensityEstimation):
    """
    Fit and evaluate multi-dimensional Kernel Density Estimation (KDE)
    using KDEpy and allowing for variable bandwidth

    Methods:
    --------
    check_dimensionality():
        check if the data matches the dimensions of KDE.

    evaluate(points):
        Evaluate the KDE at given data points.
    """
    def __init__(self, data, backend='KDEpy', bandwidth=1., dim_names=None):
        # Same initialization as parent class but default to KDEpy
        super().__init__(data, backend, bandwidth, dim_names)

    def fit_KDEpy(self):
        from KDEpy.TreeKDE import TreeKDE

        # Bandwidth may be array-like with size n_samples
        self.kernel_estimate = TreeKDE(bw=self.bandwidth).fit(self.data)

    def evaluate_KDEpy(self, points):
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
