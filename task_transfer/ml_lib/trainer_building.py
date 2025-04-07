from torch.optim import Adam

from .loss_criteria import (
    conditional_nll,
    marginal_nll,
    mc_marginal_nll,
    variational_nll,
)
from .trainer import Trainer
from .training_tools import TrainLogger


def build_flow_trainer(
    flow_model,
    data_dim,
    lr,
    weight_decay,
    eval_criterion,
    eval_params,
    eval_interval,
    early_stopping_threshold,
    early_stopping_patience,
    logging_type,
    device,
    dj_conn,
):
    """
    Build the trainer for the flow model.

    Args:
        flow_model (gensn.gensn.FlowDistribution): the flow model to be trained.

    Returns:
        Trainer: The trainer for the flow model.
    """

    loss_criterion = lambda model, batch: marginal_nll(model, batch, data_dim)
    eval_criterion = eval_criterion
    eval_params = eval_params
    optimizer = Adam(
        flow_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_logger = TrainLogger(
        model_display_name="FlowPrior",
        logging_type=logging_type,
    )
    trainer = Trainer(
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        eval_params=eval_params,
        eval_interval=eval_interval,
        optimizer=optimizer,
        lr=lr,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
        logger=train_logger,
        device=device,
        dj_conn=dj_conn,
    )
    return trainer


def build_conditional_trainer(
    model,
    data_dim,
    cond_dim,
    lr,
    weight_decay,
    eval_criterion,
    eval_params,
    eval_interval,
    early_stopping_threshold,
    early_stopping_patience,
    logging_type,
    device,
    model_display_name,
    add_eps_to_data=False,
    dj_conn=None,
):
    """
    Build the trainer for the likelihood model.

    Args:
        model (gensn.gensn.Likelihood): the likelihood model to be trained.

    Returns:
        Trainer: The trainer for the likelihood model.
    """
    loss_criterion = lambda model, batch: conditional_nll(
        model, batch, data_dim, cond_dim, add_eps=add_eps_to_data
    )
    eval_criterion = eval_criterion
    eval_params = eval_params
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_logger = TrainLogger(
        model_display_name=model_display_name,
        logging_type=logging_type,
    )
    trainer = Trainer(
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        eval_params=eval_params,
        eval_interval=eval_interval,
        optimizer=optimizer,
        lr=lr,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
        logger=train_logger,
        device=device,
        dj_conn=dj_conn,
    )
    return trainer


def build_prior_adapt_trainer(
    joint_model,
    data_dim,
    mc_sample_size,
    lr,
    weight_decay,
    eval_criterion,
    eval_params,
    eval_interval,
    early_stopping_threshold,
    early_stopping_patience,
    logging_type,
    device,
    model_display_name,
    dj_conn,
):
    loss_criterion = lambda model, batch: mc_marginal_nll(
        model, batch, data_dim, mc_sample_size
    )
    eval_criterion = eval_criterion
    eval_params = eval_params
    optimizer = Adam(
        joint_model.prior.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_logger = TrainLogger(
        model_display_name=model_display_name,
        logging_type=logging_type,
    )
    trainer = Trainer(
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        eval_params=eval_params,
        eval_interval=eval_interval,
        optimizer=optimizer,
        lr=lr,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
        logger=train_logger,
        device=device,
        dj_conn=dj_conn,
    )
    return trainer


def build_vpost_prior_trainer(
    model,
    data_dim,
    n_bound_samples,
    lr,
    weight_decay,
    eval_criterion,
    eval_params,
    eval_interval,
    early_stopping_threshold,
    early_stopping_patience,
    logging_type,
    device,
    model_display_name,
    dj_conn,
):
    # train the variational posterior and the prior
    # keep the likelihood fixed (and generally pre-trained)
    loss_criterion = lambda model, batch: variational_nll(
        model, batch, data_dim, n_bound_samples
    )
    # only pass the parameters of the variational posterior and the prior to the optimizer
    params_to_train = list(model.posterior.parameters()) + list(
        model.joint.prior.parameters()
    )  # this keeps the likelihood parameters off the optimizer
    optimizer = Adam(
        params_to_train,
        lr=lr,
        weight_decay=weight_decay,
    )
    train_logger = TrainLogger(
        model_display_name=model_display_name,
        logging_type=logging_type,
    )
    trainer = Trainer(
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        eval_params=eval_params,
        eval_interval=eval_interval,
        optimizer=optimizer,
        lr=lr,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
        logger=train_logger,
        device=device,
        dj_conn=dj_conn,
    )
    return trainer


def zero_avoid(dist_type):
    if dist_type == "gamma":
        return True
