class  SimpleGaussianKDE(object):
    """
    A class that given input values of observations
    and choices of bw  for each dimensions of input data
    compute density estimator  with bandwidth optimized 
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
        self.minX = 0 #in multi-Dimensions we need to have list with each dimenions min val
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
        leave one out cross validation 
        we need to add scipy.optimization for 
        finding best choice of bw and alpha
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

        
