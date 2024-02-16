import torch


def mc_marginal_log_likelihood(
    joint, obs, mc_sample_size=torch.Size([]), reduction="none"
):
    """
    Marginalize out the prior distribution to obtain the marginal likelihood of the data
    using Monte Carlo sampling.

    Mathematically, the marginal likelihood is given by:
    p(x) = \int p(x|z)p(z) dz
        = E_{z \sim p(z)}[p(x|z)]
        \\approx \\frac{1}{N} \sum_{i=1}^N p(x|z_i) for z_i \sim p(z)

    Args:
        joint (gensn.distributions): The joint distribution, which includes
            the prior and the conditional distributions.
        obs (torch.Tensor): The observed data of shape (batch_size, data_dim).
        mc_sample_size (torch.Size): The size of the Monte Carlo samples.
        reduction (str): The reduction method. Default is "none".
    Returns:
        torch.Tensor: The marginal log-likelihood of the observed data
            of shape (batch_size,) if reduction is "none"
            or of shape torch.Size([]) if reduction is "mean" or "sum".
    """
    prior_sample = joint.prior.rsample(mc_sample_size)
    conditional_lp = joint.conditional(obs, cond=prior_sample)
    marginal_lp = torch.logsumexp(conditional_lp, dim=0) - torch.log(
        torch.tensor(conditional_lp.shape[0])
    )
    if reduction == "none":
        ret = marginal_lp
    elif reduction == "mean":
        ret = marginal_lp.mean(0)
    elif reduction == "sum":
        ret = marginal_lp.sum(0)
    else:
        raise ValueError(f"reduction {reduction} is not supported")
    return ret
