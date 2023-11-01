# First script for iterative KDE
from awkde import GaussianKDE  #we will need our own code 


class  densityestimator(object):
    """
    A class that given input values of observations
    and choices of bw  for each dimenion of input data
    compute density estimator  with bandwoth optimized 
    using cross validation
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

        
