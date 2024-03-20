import numpy as np
import ../density_estimate
import ../utils_plot

mean1, sigma1 = 30.0, 8.0
mean2, sigma2 = 60.0, 10.0
mean3, sigma3 = 50.0, 30.0
n_samples = 100
rndgen = np.random.RandomState(seed=1)
sample1 = rndgen.normal(mean1, sigma1, size=n_samples)
print(sample1)
sample2 = rndgen.normal(mean2, sigma2, size=n_samples)
sample3 = rndgen.normal(mean3, sigma3, size=n_samples)
np.random.seed(42)
            # Number of data points
num_points = 1000
# Mean and covariance matrix
mean = [0, 0, 0]
covariance_matrix = [[1, 0.5, 0.3],
                                [0.5, 1, 0.2],
                                [0.3, 0.2, 1]]

# Generate 3D normal distributed data
sample = np.random.multivariate_normal(mean, covariance_matrix, num_points)
print(sample.shape) 
data = np.column_stack((sample1, sample2, sample3)) # shape is (n_points, n_features)
print(data.shape)
parameter = ['m1', 'm2', 'Mc'] 
kde = SimpleKernelDensityEstimation(sample,  dim_names=parameter)
fig = kde.plot_2d_contour(parameter[0],parameter[1], slice_dims=[parameter[2]], slice_values=[0], num_points=100)
