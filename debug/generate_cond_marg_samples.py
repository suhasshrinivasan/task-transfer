import pickle as pkl
from collections import OrderedDict

import gensn.distributions as G
import torch
import torch.distributions as D
from sklearn.datasets import make_spd_matrix

from task_transfer.ml_lib.loss_criteria import mc_marginal_nll
from task_transfer.utils.math_utils import is_pos_def
from task_transfer.utils.utils import dict_product


def generate_cond_marg_samples(
    prior_dim,
    conditional_dim,
    mc_sample_size,
    _seed=42,
):
    # set random seed
    torch.manual_seed(_seed)
    # construct prior and conditional of the generative model
    mu_prior = torch.rand((prior_dim,))
    cov_prior = torch.eye(prior_dim)
    prior = G.IndependentNormal(loc=mu_prior, scale=torch.sqrt(cov_prior.diag()))
    mu_conditional = torch.nn.Linear(prior_dim, conditional_dim)
    mu_conditional.weight.data = torch.rand(conditional_dim, prior_dim)
    mu_conditional.bias.data = torch.rand(conditional_dim)
    cov_conditional = torch.Tensor(make_spd_matrix(conditional_dim))
    conditional = G.TrainableDistributionAdapter(
        D.MultivariateNormal, loc=mu_conditional, covariance_matrix=cov_conditional
    )
    # construct the marginal analytically
    mu_marginal = mu_prior @ mu_conditional.weight.data.T + mu_conditional.bias.data
    marginal_cov = (
        mu_conditional.weight.data @ cov_prior @ mu_conditional.weight.data.T
        + cov_conditional
    )
    marginal = G.TrainableDistributionAdapter(
        D.MultivariateNormal, loc=mu_marginal, covariance_matrix=marginal_cov
    )
    # sample from cond
    prior_samples = prior.sample((mc_sample_size,))
    cond_samples = conditional.distribution(cond=prior_samples).sample()
    # sample from marginal
    marg_samples = marginal.sample((mc_sample_size,))

    # also compute the marginal nll

    return prior_samples, cond_samples, marg_samples


param_configs = dict_product(
    {
        "prior_dim": [1, 2, 5, 10, 50, 100, 500, 1000],
        "conditional_dim": [2],
        "mc_sample_size": [
            10_000,
        ],
        "_seed": [42],
    },
    insert_hash=False,
)

all_cond_samples = []
all_marg_samples = []
for param_config in param_configs:
    try:
        _, cond_samples, marg_samples = generate_cond_marg_samples(**param_config)
        all_cond_samples.append(cond_samples)
        all_marg_samples.append(marg_samples)
    except Exception as e:
        print("Failed config:", param_config)
        print(e)

with open("cond_samples.pkl", "wb") as f:
    pkl.dump(all_cond_samples, f)

with open("marg_samples.pkl", "wb") as f:
    pkl.dump(all_marg_samples, f)
