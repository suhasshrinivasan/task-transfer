import inspect

import gensn.distributions as G
import torch
import torch.distributions as D

from task_transfer.routines import mc_marginal_log_likelihood


class TestRoutines:
    """
    Tests for code in task_transfer/routines.py.
    """

    def test_mc_marginal_log_likelihood(
        self,
        prior_dim=10,
        conditional_dim=20,
        mc_sample_size=(10000,),
        obs_batch_dim=128,
        _seed=42,
        _atol_per_dim=1e-1,
    ):
        """
        Tests the Monte Carlo estimation of marginal log likelihood by comparing the
        estimation error against a predefined tolerance level.

        This function initializes a joint distribution composed of a prior and a conditional
        distribution, both parameterized as multivariate normal distributions. It then
        calculates the true marginal log likelihood of a set of observations directly
        and compares it against the Monte Carlo estimate of the same quantity.

        Parameters:
        - prior_dim (int): Dimensionality of the prior distribution.
        - conditional_dim (int): Dimensionality of the conditional distribution.
        - mc_sample_size (tuple): Number of samples to draw in the Monte Carlo estimation.
        - obs_batch_dim (int): The batch size of observations to evaluate.
        - _seed (int): Seed for random number generation to ensure reproducibility.
        - _atol_per_dim (float): Absolute tolerance per dimension for the error between
            the true marginal log likelihood and the Monte Carlo estimate.

        The function asserts that the error per dimension of the Monte Carlo estimate
        compared to the true marginal log likelihood is less than the specified absolute
        tolerance, indicating the estimation is sufficiently accurate.
        """
        torch.manual_seed(_seed)
        mu_prior = torch.rand((prior_dim,))
        cov_prior = torch.rand((prior_dim, prior_dim))
        cov_prior = cov_prior @ cov_prior.T + 1e-6 * torch.eye(prior_dim)
        prior = G.TrainableDistributionAdapter(
            D.MultivariateNormal, loc=mu_prior, covariance_matrix=cov_prior
        )
        mu_x_z_fn = torch.nn.Linear(prior_dim, conditional_dim)
        mu_x_z_fn.weight.data = torch.rand(conditional_dim, prior_dim)
        mu_x_z_fn.bias.data = torch.rand(conditional_dim)
        cov_x_z = torch.rand((conditional_dim, conditional_dim))
        cov_x_z = cov_x_z @ cov_x_z.T + 1e-6 * torch.eye(conditional_dim)
        conditional = G.TrainableDistributionAdapter(
            D.MultivariateNormal, loc=mu_x_z_fn, covariance_matrix=cov_x_z
        )
        joint = G.Joint(prior=prior, conditional=conditional)
        marginal_mu = mu_prior @ mu_x_z_fn.weight.data.T + mu_x_z_fn.bias.data
        marginal_cov = (
            mu_x_z_fn.weight.data @ cov_prior @ mu_x_z_fn.weight.data.T + cov_x_z
        )
        marginal = G.TrainableDistributionAdapter(
            D.MultivariateNormal, loc=marginal_mu, covariance_matrix=marginal_cov
        )
        obs = torch.rand((obs_batch_dim, conditional_dim))
        true_marginal_lp = marginal(obs).mean()
        mc_marginal_lp = mc_marginal_log_likelihood(
            joint, obs, mc_sample_size=mc_sample_size, reduction="mean"
        )
        error = torch.abs(mc_marginal_lp - true_marginal_lp) / conditional_dim
        print(
            f"Error per dimension in {inspect.currentframe().f_code.co_name}: {error}"
        )
        assert error < _atol_per_dim
