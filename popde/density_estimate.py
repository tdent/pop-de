import numpy as np
import utils_plot 

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
        Example
        --------
        # Two dimensional case
        mean1, sigma1 = 14.0, 1.5
        mean2, sigma2 = 3.0, 0.25
        n_samples = 1000
        rndgen = np.random.RandomState(seed=1)
        sample1 = rndgen.normal(mean1, sigma1, size=n_samples)
        sample2 = rndgen.normal(mean2, sigma2, size=n_samples)
        sample = np.column_stack((sample1, sample2)) # shape is (n_points, n_features)
        kde = SimpleKernelDensityEstimation(sample, dim_names=['mass1', 'mass2'])
        xgrid = np.linspace(sample1.min(), sample1.max(), 100)
        ygrid = np.linspace(sample2.min(), sample2.max(), 100)
        XX, YY = np.meshgrid(xgrid, ygrid)
        eval_pts = np.column_stack((XX.flatten(), YY.flatten()))
        zz = kde.evaluate(eval_pts)
        ZZ = zz.reshape(XX.shape)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        # Older mpl versions may require shading='flat'
        c = plt.pcolormesh(XX, YY, ZZ, cmap='Blues', norm=LogNorm(), shading='nearest')
        plt.colorbar(c)
        plt.scatter(sample1, sample2, s=2, marker='+', c='white')
        plt.xlabel(kde.dim_names[0])
        plt.ylabel(kde.dim_names[1])
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


    def plot_2d_contour(self, dim1, dim2, slice_dims=None, slice_values=None, file_name=None, num_points=100, **kwargs):
        """
        Plot a 2D contour of the KDE with optional slicing along other dimensions.

        Parameters:
            - dim1, dim2: Dimensions for the plot axes.
            - slice_dims: Dimensions to slice along (list or tuple).
            - slice_values: Values for slicing along slice_dims (list or tuple).
            - num_points: Number of points for the contour plot.
            - **kwargs: Additional keyword arguments passed to the `contour` function.
        Example:
            np.random.seed(42)
            # Number of data points
            num_points = 1000
            # Mean and covariance matrix
            mean = [0, 0, 0]
            covariance_matrix = [[1, 0.5, 0.3],
                                [0.5, 1, 0.2],
                                [0.3, 0.2, 1]]

            # Generate 3D normal distributed data
            data = np.random.multivariate_normal(mean, covariance_matrix, num_points)
            parameter = ['m1', 'm2', 'Mc']
            kde = SimpleKernelDensityEstimation(sample,  dim_names=parameter)
            # Plot a 2D contour with a slice along the 'z' dimensios
            fig = kde.plot_2d_contour(parameter[0],parameter[1], 
                    slice_dims=[parameter[2]], slice_values=[0], num_points=100)
        """
        if dim1 not in self.dim_names or dim2 not in self.dim_names:
            raise ValueError("Invalid dimension names")
        #get the samples along the dimensions 
        idx_dim1 = self.dim_names.index(dim1)
        idx_dim2 = self.dim_names.index(dim2)
        # Generate a grid for the contour plot
        xx, yy = utils_plot.get_twoD_grid(self.data[:, idx_dim1], self.data[:, idx_dim2], num_points=num_points)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        #check if slice_dimensions  == dim(KDE) -  2
        if len(slice_dims) != self.data.shape[1] - 2 :
            raise ValueError(f"With {self.data.shape[1]} KDE dimensions, must specify {self.data.shape[1] - 2} slicing parameters for the plot")

        # If slicing is specified, insert the slice values into the positions array
        if slice_values is not None:
            for slice_dim, slice_value in zip(slice_dims, slice_values):
                slice_idx = self.dim_names.index(slice_dim)
                positions = np.insert(positions, slice_idx, slice_value, axis=1)

        # Evaluate the KDE at the grid points
        z = self.evaluate_scipy(positions)

        zz = z.reshape(xx.shape)

        # Create the contour plot
        fig = utils_plot.simple2Dplot(xx, yy, zz, xlabel=dim1, ylabel=dim2, title='2D Contour Plot of KDE for {dim1} and {dim2} (Sliced along {slice_dims})')
        if file_name is not None:
            fig.savefig(file_name)
        return fig



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
