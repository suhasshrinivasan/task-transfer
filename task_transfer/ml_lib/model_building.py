import math

import gensn
import gensn.distributions as G
import torch
import torch.distributions as D
from gensn.distributions import TrainableDistributionAdapter
from gensn.flow import FlowDistribution
from gensn.transforms.invertible import Affine, IndependentAffine, SequentialTransform

from .modules import MLP, ConcRate, LocScale
from .transform_lookup import inv_nonlins


def initialize_affine_transform_(affine, init_mean, init_std=None):
    if init_std is None:
        init_std = 1.0 / math.sqrt(affine.weight.shape[0])
    affine.weight.data.normal_(mean=init_mean, std=init_std)
    # TODO: reconsider this initialization
    affine.bias.data.normal_(mean=init_mean, std=init_std)


def build_transform_sequence(
    dims=1,
    depth=5,
    nonlin="elu",
    initial_nonlin="none",
    final_nonlin="none",
    affine_init_mean=1.3,
    affine_init_std=None,
    affine_type="factorized",
):
    layers = []

    # put the first nonlinearity if given
    if initial_nonlin != "none":
        layers.append(inv_nonlins[initial_nonlin]())

    # stack sequence of affine + nonlin
    # NOTE: this assumes that sharing identical nonlin layer
    # multiple times in a network is okay. Probably fine.
    nonlin_layer = inv_nonlins[nonlin]()
    if affine_type == "factorized":
        affine_layer_class = IndependentAffine
    elif affine_type == "full":
        affine_layer_class = Affine
    elif affine_type == "lowrank":
        return NotImplementedError("Lowrank not implemented")
    else:
        raise ValueError("Unknown affine type")

    for _ in range(depth):
        affine_layer = affine_layer_class(dims)
        initialize_affine_transform_(
            affine_layer, init_mean=affine_init_mean, init_std=affine_init_std
        )
        layers.append(affine_layer)
        layers.append(nonlin_layer)

    # why is this separated?
    if depth > 0:
        affine_layer = affine_layer_class(dims)
        initialize_affine_transform_(
            affine_layer, init_mean=affine_init_mean, init_std=affine_init_std
        )
        layers.append(affine_layer)

    #  put final nonlinearity if given
    if final_nonlin != "none":
        layers.append(inv_nonlins[final_nonlin]())

    return layers


def build_flow_model(
    dims,
    flow_base_distribution,
    flow_depth,
    flow_initial_nonlinearity,
    flow_nonlinearity,
    flow_final_nonlinearity="none",
    affine_type="factorized",
):
    if flow_base_distribution == "normal":
        flow_base_distribution = G.IndependentNormal(
            loc=torch.zeros(dims), scale=torch.ones(dims)
        )
    elif flow_base_distribution == "uniform":
        flow_base_distribution = TrainableDistributionAdapter(
            G.wrap_with_indep(D.Uniform), low=torch.zeros(dims), high=torch.ones(dims)
        )
    elif flow_base_distribution == "multivariate_normal":
        # loc = torch.nn.Parameter(torch.randn(dims))
        loc = torch.zeros(dims)
        cov = gensn.parameters.Covariance(n_dims=dims)
        flow_base_distribution = G.TrainableDistributionAdapter(
            D.MultivariateNormal, loc=loc, covariance_matrix=cov
        )
    elif "lowrank_multivariate_normal" in flow_base_distribution:
        # loc = torch.nn.Parameter(torch.randn(dims))
        loc = torch.zeros(dims)
        rank = int(flow_base_distribution.split("_")[-1])
        cov = gensn.parameters.Covariance(n_dims=dims, rank=rank)
        flow_base_distribution = G.TrainableDistributionAdapter(
            D.MultivariateNormal, loc=loc, covariance_matrix=cov
        )
    # TODO: add cases for conditional normal / uniform by simple construction
    else:
        raise ValueError("Unknown base distribution for flow")

    # create the sequential flow
    flow_transform = SequentialTransform(
        *build_transform_sequence(
            dims=dims,
            depth=flow_depth,
            nonlin=flow_nonlinearity,
            initial_nonlin=flow_initial_nonlinearity,
            final_nonlin=flow_final_nonlinearity,
            affine_type=affine_type,
        )
    )

    # construct the flow distribution
    return FlowDistribution(flow_base_distribution, flow_transform)


def build_loc_scale_mlp(
    in_features,
    out_features_core,
    out_features_loc,
    out_features_scale,
    n_layers,
    nonlin,
    dropout_rate,
    init_std=1e-3,
    nonneg_transform="exp",
    clamp_pre_scale=False,
    pre_scale_max=10.0,
):
    mlp_core = MLP(
        in_features=in_features,
        out_features=out_features_core,
        n_layers=n_layers,
        nonlin=nonlin,
        dropout_rate=dropout_rate,
        init_std=init_std,
    )
    return LocScale(
        mlp_core,
        out_features_loc,
        out_features_scale,
        init_std=init_std,
        nonneg_transform=nonneg_transform,
        clamp_pre_scale=clamp_pre_scale,
        pre_scale_max=pre_scale_max,
    )


def build_conc_rate_mlp(
    in_features,
    out_features_core,
    out_features_loc,
    out_features_scale,
    n_layers,
    nonlin,
    dropout_rate,
    init_std=1e-3,
    nonneg_transform="exp",
    clamp_pre_conc=True,
    pre_conc_max=4.0,
    clamp_pre_rate=True,
    pre_rate_min=-1.6,
):
    mlp_core = MLP(
        in_features=in_features,
        out_features=out_features_core,
        n_layers=n_layers,
        nonlin=nonlin,
        dropout_rate=dropout_rate,
        init_std=init_std,
    )
    return ConcRate(
        mlp_core,
        out_features_loc,
        out_features_scale,
        init_std=init_std,
        nonneg_transform=nonneg_transform,
        clamp_pre_conc=clamp_pre_conc,
        pre_conc_max=pre_conc_max,
        clamp_pre_rate=clamp_pre_rate,
        pre_rate_min=pre_rate_min,
    )


def build_conditional(cond_dist, likelihood):
    if cond_dist == "indep_normal":
        return G.IndependentNormal(_parameters=likelihood)
    elif cond_dist == "gamma":
        return G.IndependentGamma(_parameters=likelihood)
    else:
        raise NotImplementedError("Unknown conditional distribution")
