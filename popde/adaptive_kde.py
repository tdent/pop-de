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


class AdaptiveKDEOptimization(AdaptiveBwKDE):
    """
    Optimize bandwidth and alpha by grid search using
    cross validation with a log likelihood figure of merit 
    """
    def __init__(self, data, bandwidth_options, alpha_options, weights=None, input_transf=None,
                 stdize=False, rescale=None, backend='KDEpy', bandwidth=1.0, alpha=0.0,
                 dim_names=None, do_fit=False, n_splits=2):
        self.alpha_options = alpha_options
        self.bandwidth_options = bandwidth_options
        self.n_splits = n_splits

        super().__init__(data, weights, input_transf, stdize, rescale, backend,
                         bandwidth, alpha, dim_names, do_fit)

    def loo_cv_score(self, bw_val, alpha_val):
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut() 
        fom = 0.
        for train_index, test_index in loo.split(self.kde_data):
            train_data, test_data = self.kde_data[train_index], self.kde_data[test_index]
            local_weights = None # FIX ME
            awkde = AdaptiveBwKDE(train_data, local_weights, input_transf=self.input_transf,
                                  stdize=self.stdize, rescale=self.rescale,
                                  bandwidth=bw_val, alpha=alpha_val)
            fom += np.log(awkde.evaluate(test_data))
        return fom

    def kfold_cv_score(self, bw_val, alpha_val, seed=42):
        """
        Perform k-fold cross-validation
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        fom = []
        for train_index, test_index in kf.split(self.kde_data):
            train_data, test_data = self.kde_data[train_index], self.kde_data[test_index]
            local_weights = None # FIX ME
            awkde = AdaptiveBwKDE(train_data, local_weights, input_transf=self.input_transf,
                                  stdize=self.stdize, rescale=self.rescale,
                                  bandwidth=bw_val, alpha=alpha_val)
            log_kde_eval = np.log(awkde.evaluate(test_data))
            fom.append(log_kde_eval.sum())
        return sum(fom)

    def optimize_parameters(self, method='loo_cv', fom_plot_name=None):
        best_params = {'bandwidth': None, 'alpha': None}

        # Perform grid search
        fom_grid = {}
        for bandwidth in self.bandwidth_options:
            for alpha in self.alpha_options:
                if method == 'kfold_cv':
                    fom_grid[(bandwidth, alpha)] = self.kfold_cv_score(bandwidth, alpha)
                else:
                    fom_grid[(bandwidth, alpha)] = self.loo_cv_score(bandwidth, alpha)

        optval = max(fom_grid, key=lambda k: fom_grid[k])
        optbw, optalpha = optval[0], optval[1]
        best_score = fom_grid[(optbw, optalpha)]
        # set optimized self.bandwidth and self.alpha and fit the KDE
        self.set_adaptive_parameter(optalpha, optbw)
        best_params = {'bandwidth': optbw, 'alpha': optalpha}
        
        if fom_plot_name is not None:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            for bw in self.bandwidth_options:
                fom_list = [fom_grid[(bw, al)] for al in self.alpha_options]
                ax.plot(self.alpha_options, fom_list, 
                        label='{0:.3f}'.format(float(bw)))
            ax.plot(optalpha, best_score, 'ko', linewidth=10, 
                    label=r'$\alpha={0:.3f}, bw={1:.3f}$'.format(optalpha, float(optbw)))
            ax.set_xlabel(r'$\alpha$', fontsize=18)
            ax.set_ylabel(r'$FOM$', fontsize=18)
            # add legends on top of plot in multicolumns
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25),
                            ncol=6, fancybox=True, shadow=True, fontsize=8)
            plt.tight_layout()
            plt.savefig(fom_plot_name+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()

        return best_params, best_score


class KDERescaleOptimization(AdaptiveBwKDE):
    """
     A class to optimize rescaling factors for each dimension and 
      [an optional alpha parameter]
    using Nelder-Mead optimization. 
    The optimization is conducted by minimizing a cross-validation
    score, based on log-likelihood as the figure of merit.

    Attributes:
        alpha (float): Initial alpha parameter for rescaling.
        bandwidth (float): Fixed bandwidth, default set to 1.0.
        n_splits (int): Number of splits for k-fold cross-validation.

    Methods:
        loo_cv_score(rescale_factors_alpha): 
            Computes Leave-One-Out cross-validation score 
            for given rescale factors.
        kfold_cv_score(rescale_factors_alpha, seed=42): 
            Computes k-fold cross-validation score 
            for given rescale factors.
        optimize_rescale_parameters(initial_rescale_factor, initial_alpha=0.0, method='loo_cv', fom_plot_name=None, bounds=None):
            Optimizes rescaling factors and alpha parameter.
    """
    def __init__(self, data, rescale_factors_and_alpha_array, weights=None, input_transf=None,
                 stdize=False, rescale=None, backend='KDEpy', bandwidth=1.0, alpha=0.0,
                 dim_names=None, do_fit=False, n_splits=2):
        """
        Initialize the KDERescaleOptimization instance 
        with data and optional configuration parameters.

        Args:
            data (array-like): Data to fit the KDE model.
            rescale_factors_and_alpha_array (array-like): Initial rescale factors
               and alpha parameter array.
            weights (array-like, optional): Weights for each data point.
            input_transf (function, optional): Input transformation function.
            stdize (bool, optional): Whether to standardize data.
            rescale (array-like, optional): Initial rescale factors 
               for each dimension.
            backend (str, optional): Backend to use, default is 'KDEpy'.
            bandwidth (float, optional): Fixed bandwidth value, default is 1.0.
            alpha (float, optional): Initial alpha value, default is 0.0.
            dim_names (list, optional): Dimension names.
            do_fit (bool, optional): Whether to fit the KDE upon initialization.
            n_splits (int, optional): Number of splits for k-fold 
               cross-validation, default is 2.
        """

        self.n_splits = n_splits
        #not sure I need this
        self.alpha = alpha
        self.bandwidth = bandwidth

        super().__init__(data, weights, input_transf, stdize, rescale, backend, 
                bandwidth, alpha, dim_names, do_fit)

    def loo_cv_score(self, rescale_factors_alpha):
        """
        Compute the Leave-One-Out Cross-Validation (LOO-CV) score
          based on log-likelihood.

        Args:
            rescale_factors_alpha (array-like): Array of rescale factors per
              dimension and alpha value.

        Returns:
            float: Negative sum of log-likelihood scores for LOO-CV.
        """
        alpha = rescale_factors_alpha[-1]
        rescale_factor = rescale_factors_alpha[0:len(rescale_factors_alpha)-1]

        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut() 
        fom = 0.
        for train_index, test_index in loo.split(self.kde_data):
            train_data, test_data = self.kde_data[train_index], self.kde_data[test_index]
            local_weights = None # FIX ME
            awkde = AdaptiveBwKDE(train_data, local_weights, input_transf=self.input_transf,
                                  stdize=self.stdize, rescale=rescale_factor,
                                  bandwidth=self.bandwidth, alpha=alpha)
            fom += np.log(awkde.evaluate(test_data))
        return -fom

    def kfold_cv_score(self, rescale_factors_alpha, seed=42):
        """
        Perform k-fold cross-validation
        """
        alpha= rescale_factors_alpha[-1]
        rescale_factor = rescale_factors_alpha[0:len(rescale_factors_alpha)-1]
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        fom = []
        for train_index, test_index in kf.split(self.kde_data):
            train_data, test_data = self.kde_data[train_index], self.kde_data[test_index]
            local_weights = None # FIX ME
            awkde = AdaptiveBwKDE(train_data, local_weights, input_transf=self.input_transf,
                                  stdize=self.stdize, rescale=rescale_factor,
                                  bandwidth=self.bandwidth, alpha=alpha)
            log_kde_eval = np.log(awkde.evaluate(test_data))
            fom.append(log_kde_eval.sum())
        return -sum(fom)

    def optimize_rescale_parameters(self, initial_rescale_factor, initial_alpha=0.0, method='loo_cv', fom_plot_name=None, bounds=None):

        """
        Fixed bounds on rescale factor otehrwise it is not working
        or use better initial guesses using MultiRescaling class
        """

        initial_choices = np.concatenate((initial_rescale_factor, initial_alpha))
        from scipy.optimize import minimize
        best_params = {'rescale_per_dim': None}
        # Perform Nelder-mead based Optimization
        if method == 'kfold_cv':
            result = minimize(
                    self.kfold_cv_score,        # func to minimize why negative?
                    initial_choices,             # Initial guess for the parameters
                    # args= ( )  #Additional arguments to pass to the objective function
                    method='Nelder-Mead',      # Optimization method
                    options={'disp': True}     # Display optimization progress
                    , bounds=bounds
            )
        else:
            result = minimize(
                    self.loo_cv_score,        # why negative
                    initial_rescale_choice,     
                    #args=(),  # Additional arguments to pass to the objective function
                    method='Nelder-Mead',
                    options={'disp': True} #, bounds #crucial maybe
            )

        #make self. rescale be the optimized results
        self.rescale = result.x 
        # Return the optimal parameters and the result min value
        return result.x, result.fun
            

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
        self.fom_val  = None

    def train_eval_kde(self, x, x_eval, bandwidth, alpha, ret_kde=False):
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
        kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alpha)
        kde.fit(x[:, np.newaxis])
        # Evaluate KDE at given points
        if isinstance(x_eval, (list, tuple, np.ndarray)) == False:
            y = kde.predict(x_eval)
        else:
            y = kde.predict(x_eval[:, np.newaxis])

        if ret_kde == True:
            return kde, y
        return y

    def loocv(self, bw, alpha):
        """
        we use self.data 
        Calculate likelihood FOM using leave one out cross validation for
        finding best choice of bandwidth and alpha
        """
        fom = 0.
        for i in range(len(self.data)):
            leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
            y = self.train_eval_kde(leave_one_sample, miss_sample, bw, alpha)
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
                fom[(gbw, alp)] = self.loocv(bw, alp)
        optvalues = max(fom, key=lambda k: fom[k])
        self.optbw, self.optalpha = optvalues[0], optvalues[1]
        self.fom_val = fom[(self.optbw, self.optalpha)]
        kdeval = train_eval_kde(samples, x_eval, self.optbw, self.optalpha)

        return self.fom_val, self.optbw, self.optalpha

