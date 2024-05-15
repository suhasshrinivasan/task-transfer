import math

import gensn.distributions as G
import torch
import torch.distributions as D
from gensn.distributions import TrainableDistributionAdapter
from gensn.flow import FlowDistribution
from gensn.transforms.invertible import IndependentAffine, SequentialTransform

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
):
    layers = []

    # put the first nonlinearity if given
    if initial_nonlin != "none":
        layers.append(inv_nonlins[initial_nonlin]())

    # stack sequence of affine + nonlin
    # NOTE: this assumes that sharing identical nonlin layer
    # multiple times in a network is okay. Probably fine.
    nonlin_layer = inv_nonlins[nonlin]()
    for _ in range(depth):
        affine_layer = IndependentAffine(dims)
        initialize_affine_transform_(
            affine_layer, init_mean=affine_init_mean, init_std=affine_init_std
        )
        layers.append(affine_layer)
        layers.append(nonlin_layer)

    # why is this separated?
    if depth > 0:
        affine_layer = IndependentAffine(dims)
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
):
    if flow_base_distribution == "normal":
        flow_base_distribution = G.IndependentNormal(
            loc=torch.zeros(dims), scale=torch.ones(dims)
        )
    elif flow_base_distribution == "uniform":
        flow_base_distribution = TrainableDistributionAdapter(
            G.wrap_with_indep(D.Uniform), low=torch.zeros(dims), high=torch.ones(dims)
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
        )
    )

    # construct the flow distribution
    return FlowDistribution(flow_base_distribution, flow_transform)
