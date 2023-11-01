# First script for iterative KDE
from awkde import GaussianKDE  #we will need our own code 


class iterkde(object):
    """
    a class that given choices of bw and smoothing parameters
    compute FOM and best params for KDE and provide KDE on best 
    params
    """
    def __init__(self, bw, alpha, minX, maxX):
        self.bw = bw
        self.alpha = alpha
        self.minX = 0 #in multi-Dimensions we need to have list with each dimenions min val
        self.maxX =100

    def def kde_awkde(self, x, x_grid, alpha , bandwidth, ret_kde=False):
        """Kernel Density Estimation with awkde 
        inputs:
        x = training data 
        x_grid = testing data
        alp = smoothing factor for local bw
        bandwidth = global bw for kde
        kwargs:
        ret_kde optional 
        if True kde will be output with estimated kde-values 
        """
        kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alp, diag_cov=True)
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
        if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) #bwchoice.astype(np.float) #for list
        fom = 0.0
        for i in range(len(sample)):
            leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
            y = kde_awkde(leave_one_sample, miss_sample, alp=alphachoice, gl_bandwidth=bwchoice)
            fom += np.log(y)
        return fom


