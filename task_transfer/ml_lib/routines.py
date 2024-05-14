from collections import OrderedDict
from copy import deepcopy

import torch


def mc_marginal_log_likelihood(joint, obs, mc_sample_size=(1,), reduction="none"):
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
    conditional_lp = joint.conditional(obs, cond=prior_sample.unsqueeze(1))
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


def lstsq_solution(Y, X):
    """
    Solves a least squares regression problem to find the coefficients and intercept
    using the pseudo-inverse method.

    This function calculates the best-fitting coefficients `A` and intercept `b` that minimize
    the residual sum of squares between the observed targets `Y` and the predicted targets
    based on the features `X`.

    Args:
        Y (torch.Tensor): The observed target values of shape `(n_samples,)` or `(n_samples, n_targets)`.
        X (torch.Tensor): The feature values used to predict the target, with shape `(n_samples, n_features)`.

    Returns:
        tuple: A tuple containing:
            - A (torch.Tensor): The coefficients of shape `(n_features, n_targets)`.
            - b (torch.Tensor): The intercept term(s) of shape `(1, n_targets)`.

    Raises:
        ValueError: If input matrices `X` or `Y` are empty, or if their dimensions are not compatible.

    Example:
        >>> import torch
        >>> X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        >>> Y = torch.tensor([3.0, 5.0, 7.0])
        >>> A, b = lstsq_solution(Y, X)
        >>> A, b

    Notes:
        - The solution is computed using the pseudo-inverse to handle singular or ill-conditioned matrices.
    """
    # Check if input matrices are empty
    if Y.numel() == 0 or X.numel() == 0:
        raise ValueError("Input matrices X and Y should not be empty.")

    # Ensure Y has two dimensions for consistent processing
    if Y.dim() == 1:
        Y = Y.view(-1, 1)

    # Check for dimensional compatibility
    if Y.size(0) != X.size(0):
        raise ValueError("X and Y must have the same number of rows.")

    # Append a column of ones for the intercept term
    design_matrix = torch.cat([X, torch.ones(X.size(0), 1, device=X.device)], dim=1)

    # Compute the pseudo-inverse for stability
    pseudo_inverse = torch.linalg.pinv(design_matrix)

    # Solve for the coefficients using the pseudo-inverse
    solution = pseudo_inverse @ Y

    # Extract coefficients and intercept
    A = solution[:-1]
    b = solution[-1]

    return A, b


# def lstsq_solution(Y, X):
#     design_matrix = torch.cat([X, torch.ones_like(X[:, :1])], dim=1)
#     solution = torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ Y
#     A = solution[:-1]
#     b = solution[-1]
#     return A, b


# From Neuralpredictors
def copy_model_state(model):
    """
    Given PyTorch module `model`, makes a copy of the state onto CPU.
    Args:
        model: PyTorch module to copy state dict of
    Returns:
        A copy of state dict with all tensors allocated on the CPU
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            copy_dict[k] = v.cpu() if v.is_cuda else v.clone()
        else:
            copy_dict[k] = deepcopy(v)
    return copy_dict
