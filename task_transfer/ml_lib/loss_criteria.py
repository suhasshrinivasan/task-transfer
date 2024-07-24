import torch

import wandb


def joint_nll(model, batch):
    """
    Computes the negative log-likelihood of the batch of data using the joint model.

    Args:
        model (gensn.distributions): The joint model
        batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).

    Returns:
        torch.Tensor: The negative log-likelihood of the batch of data.
    """
    response, obs = batch
    return -model(response, obs)


def marginal_nll(model, batch, data_dim):
    """
    Computes the negative log-likelihood of the batch of data using the marginal model.

    Args:
        model (gensn.distributions): The marginal model
        batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).
        data_dim (int): The dimensionality of the data.

    Returns:
        torch.Tensor: The negative log-likelihood of the batch of data.
    """
    data = batch[data_dim]
    return -model(data)


def conditional_nll(model, batch, data_dim, cond_dim, add_eps=False):
    """
    Computes the negative log-likelihood of the batch of data using the conditional model.

    Args:
        model (gensn.distributions): The conditional model
        batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).
        data_dim (int): The dimensionality of the data.
        cond_dim (int): The dimensionality of the conditioning variable.

    Returns:
        torch.Tensor: The negative log-likelihood of the batch of data.
    """
    data = batch[data_dim]
    cond = batch[cond_dim]
    if add_eps:
        print("!!!Warning!!! adding eps to data")
        data = data + torch.finfo(data.dtype).eps
    nll = -model(data, cond=cond)
    return nll


# TODO: self written. use this!
def mc_marginal_nll(model, batch, data_dim, mc_sample_size=(1,)):
    """
    Marginalize out the prior distribution to obtain the marginal negative log-likelihood of the data
    using Monte Carlo sampling.

    Mathematically, the marginal likelihood is given by:
    p(x) = \int p(x|z)p(z) dz
        = E_{z \sim p(z)}[p(x|z)]
        \\approx \\frac{1}{N} \sum_{i=1}^N p(x|z_i) for z_i \sim p(z)

    Args:
        model (gensn.distributions): The joint model, which includes
            the prior and the conditional distributions.
        batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).
        data_dim (int): The dimensionality of the data.
        mc_sample_size (torch.Size): The size of the Monte Carlo samples.
        reduction (str): The reduction method. Default is "none".

    Returns:
        torch.Tensor: The marginal negative log-likelihood of the batch of data
            of shape (batch_size,)
    """
    data = batch[data_dim]
    prior_sample = model.prior.rsample(mc_sample_size)
    conditional_ll = model.conditional(data, cond=prior_sample.unsqueeze(1))
    marginal_ll = torch.logsumexp(conditional_ll, dim=0) - torch.log(
        torch.tensor(conditional_ll.shape[0])
    )
    return -marginal_ll


