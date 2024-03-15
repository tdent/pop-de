import numpy as np
import transform_utils as transf


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
    def __init__(self, data, input_transf=None, stdize=False, rescale=None,
                 backend='scipy', bandwidth=1., dim_names=None):
        """
        data: array-like, shape (n_samples, n_features)
            Data points defining kernel positions
            Each row is a point, each column is a parameter.
        kwargs:
            input_transf : None or sequence of strings, eg ('log', 'none', 'log')
              describing transformations of data before KDE calculation
            stdize : Boolean, whether to standardize all data dimensions
            rescale : None or sequence of float, rescaling of dimensions immediately
              before KDE calculation
            backend : String, Processing method to do KDE calculation
            bandwidth : Float or array of float, bandwidth of kernels used for smoothing
            dim_names : Sequence of dimension names, e.g. ('m1', 'z', 'chi_eff')

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
        self.ndim = data.shape[1]

        self.data = np.asarray(data)
        self.input_transf = input_transf
        self.stdize = stdize
        self.rescale = rescale

        self.backend = backend
        self.bandwidth = bandwidth
        self.dim_names = dim_names
        if dim_names is not None:
            self.check_dimensionality()

        # Do transformation, standardize and rescale input data
        self.prepare_data()

        self.kernel_estimate = None
        # Initialize the KDE
        self.fit()

    def check_dimensionality(self):
        """
        Check if the dimension of training data matches the number of param names
        and data preparation option values
        """
        if self.dim_names is not None:
            if len(self.dim_names) != self.ndim:
                raise ValueError("Dimensionality of data array does not match "
                                 "the number of dimension names.")
        if self.input_transf is not None:
            if len(self.input_transf) != self.ndim:
                raise ValueError("Dimensionality of data array does not match "
                                 "the number of transformation strings.")
        if self.rescale is not None:
            if len(self.rescale) != self.ndim:
                raise ValueError("Dimensionality of data array does not match "
                                 "the number of rescaling factors.")

    def prepare_data(self):
        """
        Transform, standardize and rescale input data into KDE-ready data
        """
        if self.input_transf is not None:
            self.transf_data = transf.transform_data(self.data, self.input_transf)
        else:
            self.transf_data = self.data

        if self.stdize:
            std_transf = ['stdize' for dim in self.ndim]
            self.stds = np.std(self.transf_data, axis=0)  # record the stds
            self.std_data = transf.transform_data(self.transf_data, std_transf)
        else:
            self.stds = None
            self.std_data = self.transf_data

        if self.rescale is not None:
            self.kde_data = transf.transform_data(self.std_data, self.rescale)
        else:
            self.kde_data = self.std_data

    def fit(self):
        """
        General fit method allowing for different backends
        """
        fit_method = getattr(self, 'fit_' + self.backend)
        fit_method()

    def fit_scipy(self):
        from scipy.stats import gaussian_kde

        # scipy takes data with shape (n_dimensions, n_samples)
        self.kernel_estimate = gaussian_kde(self.kde_data.T, bw_method=self.bandwidth)

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
    def __init__(self, data, input_transf=None, stdize=False, rescale=None,
                 backend='KDEpy', bandwidth=1., dim_names=None):
        # Same initialization as parent class but default to KDEpy
        super().__init__(data, input_transf, stdize, rescale,
                         backend, bandwidth, dim_names)  # Arguments stay in same order

    def fit_KDEpy(self):
        from KDEpy.TreeKDE import TreeKDE

        # Bandwidth may be array-like with size n_samples
        self.kernel_estimate = TreeKDE(bw=self.bandwidth).fit(self.kde_data)

    def evaluate_KDEpy(self, points):
        density_values = self.kernel_estimate.evaluate(points)

        return density_values

