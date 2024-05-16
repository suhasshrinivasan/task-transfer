def joint_nll(joint_model, batch):
    """
    Computes the negative log-likelihood of the batch of data using the joint model.

    Args:
        joint_model (gensn.distributions): The joint model, which includes
            the prior and the conditional distributions.
        batch (torch.Tensor): The batch of data of shape (batch_size, data_dim).

    Returns:
        torch.Tensor: The negative log-likelihood of the batch of data.
    """
    response, obs = batch
    return -joint_model.log_prob(response, obs)
