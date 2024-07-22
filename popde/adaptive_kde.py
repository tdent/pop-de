import numpy as np
import scipy
from density_estimate import VariableBwKDEPy 
from scipy.stats import gmean
import matplotlib.pyplot as plt


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

        # Set up initial KDE with fixed bandwidth
        super().__init__(data, weights, input_transf, stdize,
                         rescale, backend, bandwidth, dim_names, do_fit)
        # Compute pilot kde values at input points
        self.pilot_values = self.evaluate(self.kde_data)
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
        Calculate per-point bandwidths and re-initialize KDE

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
        # Update local bandwidths (pilot values remain unchanged)
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
        self.set_bandwidth(new_global_bw)
        # Update pilot_values using new global bandwidth
        self.pilot_values = self.evaluate(self.kde_data)
        
        # Set alpha and calculate per-point bandwidths 
        self.set_alpha(new_alpha)


class KDEOptimization(AdaptiveBwKDE):
    """
    Optimize bandwidth and alpha by grid search using
    cross validation with a log likelihood figure of merit 
    """
    def __init__(self, data, weights, bandwidth_options, alpha_options , input_transf=None, stdize=False, rescale=None, backend='KDEpy', bandwidth=0.5, alpha=0.0, dim_names=None, do_fit=True):
        self.alpha_options = alpha_options
        self.bandwidth_options = bandwidth_options
        super().__init__(data, weights, input_transf, stdize,
                         rescale, backend, bandwidth, alpha, dim_names, do_fit)

    def loo_cv_score(self, bandwidth_val, alpha_val):
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut() #sklearn way
        fom = 0.0
        for train_index, test_index in loo.split(self.data):
            train_data, test_data = self.data[train_index], self.data[test_index]
           
            #######There may need some hints how to use it better 
            ### also if we need to add weight option here
            local_weights = None
            awkde = AdaptiveBwKDE(train_data, local_weights, bandwidth=bandwidth_val,alpha=alpha_val)
            fom += awkde.evaluate(test_data)  
        return np.mean(fom)

    def kfold_cv_score(self, bandwidth_val, alpha_val, n_splits=2):
        """
        to do: how to choose and use s_split in init
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fom = []
        for train_index, test_index in kf.split(self.data):
            train_data, test_data = self.data[train_index], self.data[test_index]
            local_weights = None
            awkde = AdaptiveBwKDE(train_data, local_weights, bandwidth=bandwidth_val,alpha=alpha_val)
            fom.append(awkde.evaluate(test_data).sum())
        return sum(fom)

    def optimize_parameters(self, method='loo_cv', fom_plot=False):
        import operator
        #best_score = float('inf')
        best_params = {'bandwidth': None, 'alpha': None}

        FOM= {}
        for bandwidth in self.bandwidth_options:
            for alpha in self.alpha_options:
                if method=='kfold_cv':
                    score = self.kfold_cv_score(bandwidth, alpha, n_splits=2)
                else:
                    score = self.loo_cv_score(bandwidth, alpha)

                FOM[(bandwidth, alpha)] = score
                #if score < best_score:
                #    best_score = score
                #    best_params['bandwidth'] = bandwidth
                #    best_params['alpha'] = alpha

        optval = max(FOM.items(), key=operator.itemgetter(1))[0]
        optbw, optalpha  = optval[0], optval[1]
        maxFOM = FOM[(optbw, optalpha)]

        
        if fom_plot==True:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            for bw in self.bandwidth_options:
                FOMlist = [FOM[(bw, al)] for al in self.alpha_options]
                if bw not in ['silverman', 'scott']:
                    bw = float(bw) #bwchoice.astype(np.float) #for list
                    ax.plot(self.alpha_options, FOMlist, label='{0:.3f}'.format(bw))
                else:
                    ax.plot(alphagrid, FOMlist, label='{}'.format(bw))
            if optbw not in ['silverman', 'scott']:
                ax.plot(optalpha, maxFOM, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1:.3f}$'.format(optalpha, optbw))
            else:
                ax.plot(optalpha, maxFOM, 'ko', linewidth=10, label=r'$\alpha={0:.3f}, bw= {1}$'.format(optalpha, optbw))
            ax.set_xlabel(r'$\alpha$', fontsize=18)
            ax.set_ylabel(r'$FOM$', fontsize=18)
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.25), ncol =6, fancybox=True, shadow=True, fontsize=8)
        #plt.ylim(maxFOM -5 , maxFOM +6)
            plt.savefig("FOMfortwoDsourcecase.png", bbox_extra_artists=(lgd, ), bbox_inches='tight')
            plt.close()

        #set self bandwidth  and alpha
        self.bandwidth  = optbw
        self.alpha  = optalpha
        best_params = {'bandwidth': optbw, 'alpha': optalpha}

        return  best_params, maxFOM #best_params, best_score


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
        import operator
        optvalues = max(fom.items(), key=operator.itemgetter(1))[0]
        self.optbw, self.optalpha = optvalues[0], optvalues[1]
        self.fom_val = fom[(self.optbw, self.optalpha)]
        kdeval = train_eval_kde(samples, x_eval, self.optbw, self.optalpha)

        return self.fom_val, self.optbw, self.optalpha

