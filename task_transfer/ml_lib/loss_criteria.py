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
    # TODO: debug code. cleanup
    # data_hist = wandb.Histogram(data.detach().cpu())
    # cond_hist = wandb.Histogram(cond.detach().cpu())
    # wandb.log({"data/hist": data_hist, "cond/hist": cond_hist})
    # wandb.log({"data/min": data.min(), "data/max": data.max()})
    # wandb.log({"cond/min": cond.min(), "cond/max": cond.max()})
    if add_eps:
        print("!!!Warning!!! adding eps to data")
        data = data + torch.finfo(data.dtype).eps
    nll = -model(data, cond=cond)
    return nll


# # TODO: CAREFUL! ChatGPT corrected!
# def mc_marginal_nll(model, batch, data_dim, mc_sample_size=1, reduction="none"):
#     """
#     Marginalize out the prior distribution to obtain the marginal negative log-likelihood of the data
#     using Monte Carlo sampling.

#     Mathematically, the marginal likelihood is given by:
#     p(x) = \int p(x|z)p(z) dz
#         = E_{z \sim p(z)}[p(x|z)]
#         \\approx \\frac{1}{N} \sum_{i=1}^N p(x|z_i) for z_i \sim p(z)

#     Args:
#         model (gensn.distributions): The joint model, which includes
#             the prior and the conditional distributions.
#         batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).
#         data_dim (int): The dimensionality of the data.
#         mc_sample_size (int): The number of Monte Carlo samples.
#         reduction (str): The reduction method. Default is "none".

#     Returns:
#         torch.Tensor: The marginal negative log-likelihood of the batch of data
#             of shape (batch_size,) if reduction is "none"
#             or of shape torch.Size([]) if reduction is "mean" or "sum".
#     """
#     data = batch[data_dim]
#     batch_size = data.shape[0]

#     # Sample from the prior
#     prior_samples = model.prior.rsample((mc_sample_size[0], batch_size))

#     # Compute conditional log-likelihoods
#     conditional_ll = model.conditional(data, cond=prior_samples)

#     # Log-sum-exp trick for numerical stability
#     log_prob = torch.logsumexp(conditional_ll, dim=0) - torch.log(
#         torch.tensor(mc_sample_size).to(data.device)
#     )

#     # Negative log-likelihood
#     marginal_nll = -log_prob

#     if reduction == "mean":
#         return marginal_nll.mean()
#     elif reduction == "sum":
#         return marginal_nll.sum()
#     else:
#         return marginal_nll


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
            of shape (batch_size,) if reduction is "none"
            or of shape torch.Size([]) if reduction is "mean" or "sum".
    """
    data = batch[data_dim]
    prior_sample = model.prior.rsample(mc_sample_size)
    conditional_ll = model.conditional(data, cond=prior_sample.unsqueeze(1))
    # TODO: debug code. cleanup
    # wandb.log({"eval/cond_ll_sample_mean": conditional_ll.mean()})
    marginal_ll = torch.logsumexp(conditional_ll, dim=0) - torch.log(
        torch.tensor(conditional_ll.shape[0])
    )
    # TODO: debug code. cleanup
    # cond_dim = 0
    # cond = batch[cond_dim]
    # conditional_ll_real = model.conditional(data, cond=cond.unsqueeze(1))
    # wandb.log({"eval/cond_ll_real_mean": conditional_ll_real.mean()})
    # marginal_ll_real = torch.logsumexp(conditional_ll_real, dim=0) - torch.log(
    # torch.tensor(conditional_ll_real.shape[0])
    # )
    # wandb.log({"eval/marg_ll_sample": marginal_ll.mean()})
    # wandb.log({"eval/marg_ll_real": marginal_ll_real.mean()})

    return -marginal_ll


# def
