import torch
from torch.optim import Adam

from .loss_criteria import marginal_nll
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
        model_display_name="GenerativeModel",
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
