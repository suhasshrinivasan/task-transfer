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


def conditional_nll(model, batch, data_dim, cond_dim):
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
    return -model(data, cond=cond)
