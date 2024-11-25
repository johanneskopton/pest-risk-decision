import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy



n_feature_dims = 3
class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, n_covars):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(n_covars, 5))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(5, n_feature_dims))
        self.add_module('sigmoid', torch.nn.Sigmoid())


class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points, n_covars):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        
        self.feature_extractor = FeatureExtractor(n_covars)
        self.mean_module = gpytorch.means.ConstantMean()

        space_local_rbf = gpytorch.kernels.RBFKernel(active_dims=(0, 1))
        space_local_rbf.lengthscale = 0.2 # 0.2 degrees
        space_local_kernel = gpytorch.kernels.ScaleKernel(space_local_rbf)

        time_local_rbf = gpytorch.kernels.RBFKernel(active_dims=(2))
        time_local_rbf.lengthscale = 21/365 # 3 weeks
        time_local_kernel = gpytorch.kernels.ScaleKernel(time_local_rbf)

        metric_space_time_rbf = gpytorch.kernels.RBFKernel(active_dims=(0, 1, 2), ard_num_dims=3)
        metric_space_time_rbf.lengthscale = 0.1 # 5 % of total range
        metric_space_time_kernel = gpytorch.kernels.ScaleKernel(metric_space_time_rbf)

        sum_metric_kernel = space_local_kernel + time_local_kernel + metric_space_time_kernel

        time_periodic_component = gpytorch.kernels.PeriodicKernel(active_dims=(2))
        time_periodic_component.period_length = 1 # 1 year
        time_periodic_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=(2)) \
                             * time_periodic_component)


        feature_dims = tuple(range(3, 3+n_feature_dims))

        covariate_features_rbf = gpytorch.kernels.RBFKernel(active_dims=feature_dims, ard_num_dims=n_feature_dims)
        covariate_features_kernel = gpytorch.kernels.ScaleKernel(covariate_features_rbf)

        self.covar_module = sum_metric_kernel + time_periodic_kernel + covariate_features_kernel

        self.feature_extractor = self.feature_extractor


    def forward(self, x):
        x_spacetime = x[:, :3]
        x_covariates = x[:, 3:]

        x_covariates_features = self.feature_extractor(x_covariates)
        x_features = torch.cat((x_spacetime, x_covariates_features), dim=1)

        mean_x = self.mean_module(x_features)
        covar_x = self.covar_module(x_features)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred