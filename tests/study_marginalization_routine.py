import json
from collections import OrderedDict

import gensn.distributions as G
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from sklearn.datasets import make_spd_matrix

import experiments.orientation_discrimination.haefner_model.configs as data_cfg
from task_transfer.ml_lib.loss_criteria import mc_marginal_nll
from task_transfer.utils.insilico_stimuli import generate_gabors
from task_transfer.utils.math_utils import is_pos_def
from task_transfer.utils.model_utils import build_haefner_model
from task_transfer.utils.utils import dict_product


def test_mc_marginal_log_likelihood(
    prior_dim=10,
    prior_var=1,
    conditional_dim=144,
    conditional_var=1,
    mc_sample_size=10_000,
    _sampling_seed=42,
    _model_seed=42,
    make_pfs_ortho=False,
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
    - mc_sample_size (int): Number of samples to draw in the Monte Carlo estimation.
    - obs_batch_dim (int): The batch size of observations to evaluate.
    - _seed (int): Seed for random number generation to ensure reproducibility.
    - _atol_per_dim (float): Absolute tolerance per dimension for the error between
        the true marginal log likelihood and the Monte Carlo estimate.

    The function asserts that the error per dimension of the Monte Carlo estimate
    compared to the true marginal log likelihood is less than the specified absolute
    tolerance, indicating the estimation is sufficiently accurate.
    """
    print("Running config:", locals())
    torch.manual_seed(_model_seed)
    mc_sample_size = (mc_sample_size,)
    mu_prior = torch.rand((prior_dim,))
    # cov_prior = torch.rand((prior_dim, prior_dim))
    # cov_prior = cov_prior @ cov_prior.T + 1e-6 * torch.eye(prior_dim)
    # cov_prior = torch.Tensor(make_spd_matrix(prior_dim))
    # cov_prior = torch.eye(prior_dim) * torch.randn(prior_dim) ** 2
    # prior = G.TrainableDistributionAdapter(
    #     D.MultivariateNormal, loc=mu_prior, covariance_matrix=cov_prior
    # )
    cov_prior = torch.eye(prior_dim) * prior_var
    prior = G.IndependentNormal(loc=mu_prior, scale=torch.sqrt(cov_prior.diag()))
    mu_x_z_fn = torch.nn.Linear(prior_dim, conditional_dim)

    x_phi = torch.linspace(0, np.pi, steps=prior_dim + 1)[:-1]
    canvas_dim = int(np.sqrt(conditional_dim))
    gabor_params = dict(
        {
            "canvas_size": [canvas_dim, canvas_dim],
            "sizes": [canvas_dim],
            "spatial_frequencies": [1 / 3],
            "contrasts": [1.0],
            "grey_levels": [0.0],
            "eccentricities": [0.0],
            "locations": [[canvas_dim // 2, canvas_dim // 2]],
            "phases": [np.pi / 2],
            "relative_sf": False,
        },
    )
    x_pfs = torch.Tensor(
        generate_gabors(orientations=x_phi.tolist(), gabor_params=gabor_params)
    )
    x_pfs = x_pfs.view(x_pfs.shape[0], -1).T
    if make_pfs_ortho:
        torch.cat(
            [
                torch.diag(torch.diag(x_pfs)),
                torch.zeros(conditional_dim - prior_dim, prior_dim),
            ],
            dim=0,
        ).shape
    mu_x_z_fn.weight.data = x_pfs
    mu_x_z_fn.bias.data = torch.zeros(conditional_dim)

    # cov_x_z = torch.Tensor(make_spd_matrix(conditional_dim))
    cov_x_z = torch.eye(conditional_dim) * conditional_var
    # C = torch.randn((conditional_dim, conditional_dim)) / 10
    # L = torch.tril(C)
    # cov_x_z = L @ L.T + 1e-4 * torch.eye(conditional_dim)
    # cov_x_z = cov_x_z @ cov_x_z.T + 1e-4 * torch.eye(conditional_dim)
    conditional = G.TrainableDistributionAdapter(
        D.MultivariateNormal, loc=mu_x_z_fn, covariance_matrix=cov_x_z
    )
    joint = G.Joint(prior=prior, conditional=conditional)
    marginal_mu = mu_prior @ mu_x_z_fn.weight.data.T + mu_x_z_fn.bias.data
    marginal_cov = mu_x_z_fn.weight.data @ cov_prior @ mu_x_z_fn.weight.data.T + cov_x_z
    marginal = G.TrainableDistributionAdapter(
        D.MultivariateNormal, loc=marginal_mu, covariance_matrix=marginal_cov
    )
    # obs = marginal.sample((obs_batch_dim,))  # TODO: make mean of marginal
    obs = marginal_mu.unsqueeze(0)
    true_marginal_lp = marginal(obs).mean()
    # mc_marginal_lp = mc_marginal_log_likelihood(
    #     joint, obs, mc_sample_size=mc_sample_size, reduction="mean"
    # )
    # data = obs
    # prior_sample = joint.prior.rsample(mc_sample_size)
    # conditional_dist = joint.conditional.distribution(cond=prior_sample.unsqueeze(1))
    # conditional_ll = conditional_dist.log_prob(data)
    # marginal_ll = torch.logsumexp(conditional_ll, dim=0) - torch.log(
    #     torch.tensor(conditional_ll.shape[0])
    # )
    # marginal_nll, prior_sample, conditional_dist, conditional_ll = (
    #     mc_marginal_nll_detailed(joint, [obs, obs], 0, mc_sample_size)
    # )
    torch.manual_seed(_sampling_seed)
    marginal_lp = -mc_marginal_nll(
        joint, [obs, None], 0, mc_sample_size=mc_sample_size
    ).mean()
    error = true_marginal_lp - marginal_lp
    print("Error:", error.item())
    abs_error = torch.abs(error)
    mse = torch.mean(abs_error**2)
    return error.item(), abs_error.item(), mse.item()


# param_configs = dict_product(
#     {
#         "prior_dim": [1, 5, 10, 50, 100, 500, 1000],
#         "conditional_dim": [1, 5, 10, 50, 100, 500, 1000],
#         "mc_sample_size": [
#             (1,),
#             (5,),
#             (10,),
#             (50,),
#             (100,),
#             (1000,),
#         ],
#         "obs_batch_dim": [1, 5, 10, 50, 100],
#         "_seed": [42],
#     },
#     insert_hash=False,
# )

# param_configs = dict_product(
#     {
#         "prior_dim": [
#             1,
#             2,
#             5,
#             10,
#             50,
#             100,
#             500,
#             1000,
#         ],
#         "conditional_dim": [2],
#         "mc_sample_size": [(10_000,)],
#         "obs_batch_dim": [128],
#         "_seed": [42],
#     },
#     insert_hash=False,
# )

# param_configs = dict_product(
#     {
#         "prior_dim": range(1, 100, 5),
#         "conditional_dim": range(1, 100, 5),
#         "mc_sample_size": [
#             (10000,),
#         ],
#         "obs_batch_dim": [128],
#         "_seed": [42],
#     },
#     insert_hash=False,
# )

# param_configs = dict_product(
#     {
#         "prior_dim": [45],
#         "conditional_dim": [144],
#         "mc_sample_size": [
#             (1,),
#             (5,),
#             (10,),
#             (50,),
#             (100,),
#             (500,),
#             (1000,),
#             (5_000,),
#             (10_000,),
#             (50_000,),
#             # (100_000,),
#         ],
#         "obs_batch_dim": [128],
#         "_seed": [42, 23432, 5745, 196],
#     },
#     insert_hash=False,
# )

# param_configs = dict_product(
#     {
#         "prior_dim": [45],
#         "conditional_dim": [144],
#         "mc_sample_size": range(1, 50_000, 10),
#         "obs_batch_dim": [128],
#         "_seed": [42, 23432, 5745, 196],
#     },
#     insert_hash=False,
# )

# param_configs = dict_product(
#     {
#         "prior_dim": range(1, 45),
#         "conditional_dim": [144],
#         "mc_sample_size": [10, 100, 1000, 10_000],
#         "obs_batch_dim": [128],
#         "_seed": [42],
#     },
#     insert_hash=False,
# )

param_configs = dict_product(
    {
        "prior_dim": range(1, 46, 4),
        "prior_var": [1e-1, 5e-1, 1],
        "conditional_dim": [36],
        "conditional_var": [1e-1, 5e-1, 1],
        "mc_sample_size": [100, 1000, 10_000],
        "_sampling_seed": range(100),
        "_model_seed": [42],
        "make_pfs_ortho": [False],
    },
    insert_hash=False,
)


prior_dims = []
prior_vars = []
conditional_dims = []
conditional_vars = []
mc_sample_sizes = []
sampling_seeds = []
model_seeds = []
errors = []
abs_errors = []
mses = []
for param_config in param_configs:
    try:
        error, abs_error, mse = test_mc_marginal_log_likelihood(**param_config)
        errors.append(error)
        abs_errors.append(abs_error)
        mses.append(mse)
        prior_dims.append(param_config["prior_dim"])
        prior_vars.append(param_config["prior_var"])
        conditional_dims.append(param_config["conditional_dim"])
        conditional_vars.append(param_config["conditional_var"])
        mc_sample_sizes.append(param_config["mc_sample_size"])
        sampling_seeds.append(param_config["_sampling_seed"])
        model_seeds.append(param_config["_model_seed"])
    except Exception as e:
        print("Failed config:", param_config)
        print(e)

result = OrderedDict(
    {
        "errors": errors,
        "abs_errors": abs_errors,
        "mses": mses,
        "prior_dim": prior_dims,
        "prior_var": prior_vars,
        "conditional_dim": conditional_dims,
        "conditional_var": conditional_vars,
        "mc_sample_size": mc_sample_sizes,
        "sampling_seed": sampling_seeds,
        "model_seed": model_seeds,
    }
)
with open("mc_error_comprehensive.json", "w") as f:
    json.dump(result, f, indent=2)
