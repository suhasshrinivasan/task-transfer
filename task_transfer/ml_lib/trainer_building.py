import torch
from torch.optim import Adam

from .loss_criteria import conditional_nll, marginal_nll, mc_marginal_nll
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
):
    """
    Build the trainer for the likelihood model.

    Args:
        model (gensn.gensn.Likelihood): the likelihood model to be trained.

    Returns:
        Trainer: The trainer for the likelihood model.
    """

    loss_criterion = lambda model, batch: conditional_nll(
        model, batch, data_dim, cond_dim
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
    )
    return trainer
