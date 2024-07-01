import numpy as np
import transform_utils as transf
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
        self.ndim = self.data.shape[1]
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
            std_transf = ['stdize'] * self.ndim
            self.stds = np.std(self.transf_data, axis=0)  # record the stds
            self.std_data = transf.transform_data(self.transf_data, std_transf)
        else:
            self.std_data = self.transf_data

        if self.rescale is not None:
            self.kde_data = transf.transform_data(self.std_data, self.rescale)
        else:
            self.kde_data = self.std_data


    def evaluate_with_transf(self, points):
        """
        Transforms the input points in the same way as the KDE training data,
        evaluates the KDE on the transformed points and adjusts the KDE
        values by the Jacobian of the transformations.

        Parameters:
        -----------
        points : array-like, shape (n_samples, n_features)
            The original parameter space points to be transformed and evaluated.

        Returns:
        --------
        kde_vals : array-like, shape (n_samples,)
        The KDE values adjusted by the Jacobian of the transformations.
        """
        if self.input_transf is not None:
            transf_points = transf.transform_data(points, self.input_transf)
        else:
            transf_points = points

        # Divide each parameter by the std of the training data
        if self.stdize:
            std_points = transf.transform_data(transf_points, 1.0/self.stds)
        else:
            std_points = transf_points

        # Rescale each parameter
        if self.rescale is not None:
            transf_data = transf.transform_data(std_points, self.rescale)       
        else:
            transf_data = std_points

        #Evaluate kde on transform points
        kde_vals = self.evaluate(transf_data)

        # Jacobian of transform
        for i, option in enumerate(self.input_transf):
            if option in['log', 'ln']:
                input_Jacobian = 1.0 / points[:, i]
            elif option == 'exp':
                input_Jacobian = np.exp(points[:, i])
            elif option in ('none', 'None'):
                input_Jacobian  = 1.0 
            else:
                raise ValueError(f"Invalid transformation option at index {i}: {option}")

            #standardize using stds of training data 
            if self.stdize:
                Jacobian_std_factor = self.stds[i]
            else:
                Jacobian_std_factor = 1.0

            #rescale based on self.rescale list in each dimension
            if self.rescale is not None:
                Jacobian_rescale_factor = self.rescale[i]
            else:
                Jacobian_rescale_factor = 1.0

            Total_Jacobian =  input_Jacobian * Jacobian_std_factor * Jacobian_rescale_factor
            kde_vals   *= Total_Jacobian

        return kde_vals


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

        Parameters
        ----------
        points : array-like
            The parameter points at which the KDE will be evaluated

        Returns
        -------
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

    def plot_2d_contour(self, dim1, dim2, slice_dims=None, slice_values=None, num_points=100, file_name=None, **kwargs):
        """
        Plot a 2D contour of the KDE with optional slicing along other dimensions.

        Parameters
        ----------
        dim1, dim2 : string
            Dimension names for the plot axes.
        slice_dims : iterable of string
            Dimensions to slice along.
        slice_values : iterable of float
            Values for slicing along slice_dims.
        num_points : int
            Number of grid points to evaluate along each axis.
        file_name : string
            File to save the plot in.

        Returns
        -------
        fig : matplotlib.Figure instance
            Plot handle       

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
        # Input checking
        if dim1 not in self.dim_names or dim2 not in self.dim_names:
            raise ValueError("Invalid dimension names")
    
        if len(slice_dims) != self.data.shape[1] - 2:
            raise ValueError(f"With {self.data.shape[1]} KDE dimensions, must specify {self.data.shape[1] - 2} slicing parameters for the plot")

        if len(slice_dims) != len(slice_values):
            raise ValueError(f"Number of slice dimensions must match number of slice values")

        # Find the KDE dimensions to plot
        idx_dim1 = self.dim_names.index(dim1)
        idx_dim2 = self.dim_names.index(dim2)

        # Generate a grid for the contour plot
        xx, yy = utils_plot.get_twoD_grid(self.data[:, idx_dim1], self.data[:, idx_dim2], num_points=num_points)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        
        # If slicing is specified, insert the slice values into the positions array
        if slice_values is not None:
            for slice_dim, slice_value in zip(slice_dims, slice_values):
                slice_idx = self.dim_names.index(slice_dim)
                positions = np.insert(positions, slice_idx, slice_value, axis=1)

        # Evaluate the KDE at the grid points
        z = self.evaluate(positions)

        # Create the contour plot
        zz = z.reshape(xx.shape)
        fig = utils_plot.simple2Dplot(xx, yy, zz, xlabel=dim1, ylabel=dim2, title=f'KDE sliced along {slice_dims} at {slice_values})')

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

