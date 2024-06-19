import inspect

import torch

import wandb
from task_transfer.evaluation.evaluate_generative_model import (
    compute_logl,
    logl_mc_marginal,
)
from task_transfer.ml_lib.data_loading import build_dataloaders
from task_transfer.ml_lib.model_building import build_joint_model
from task_transfer.ml_lib.trainer_building import build_prior_adapt_trainer


def adapt_prior(data_loader_args, model_args, trainer_args, use_wandb=False):
    torch.manual_seed(model_args["seed"])
    current_function_name = inspect.currentframe().f_code.co_name
    if use_wandb:
        wandb.init(
            project=current_function_name,
            entity="walkerlab",
            config={**data_loader_args, **model_args, **trainer_args},
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        data_fname=data_loader_args["data_fname"],
        train_prop=data_loader_args["train_prop"],
        val_prop=data_loader_args["val_prop"],
        batch_size=trainer_args["batch_size"],
    )

    prior_model = torch.load(
        model_args["prior_model_path"], map_location=trainer_args["device"]
    )
    conditional_model = torch.load(
        model_args["likelihood_model_path"], map_location=trainer_args["device"]
    )
    joint_model = build_joint_model(prior_model, conditional_model)

    image_dim = 1
    response_dim = 0
    trainer = build_prior_adapt_trainer(
        joint_model=joint_model,
        data_dim=image_dim,
        mc_sample_size=(trainer_args["mc_sample_size"],),
        lr=trainer_args["lr"],
        weight_decay=trainer_args["weight_decay"],
        eval_criterion=None,
        eval_params=None,
        eval_interval=None,
        early_stopping_threshold=trainer_args["early_stopping_threshold"],
        early_stopping_patience=trainer_args["early_stopping_patience"],
        logging_type="wandb" if use_wandb else "stdout",
        device=trainer_args["device"],
        model_display_name=current_function_name,
    )

    trainer_output = trainer.train(
        model=joint_model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=trainer_args["n_epochs"],
    )

    tracker_output = trainer_output["tracker_output"]
    eval_output = trainer_output["eval_output"]

    eval_mc_sample_size = (1000,)  # TODO: parameterize this?
    train_marginal_obs_ll_mean, train_marginal_obs_ll_sem = logl_mc_marginal(
        joint_model,
        train_loader,
        data_dim=image_dim,
        mc_sample_size=eval_mc_sample_size,
    )

    val_marginal_obs_ll_mean, val_marginal_obs_ll_sem = logl_mc_marginal(
        joint_model,
        val_loader,
        data_dim=image_dim,
        mc_sample_size=eval_mc_sample_size,
    )

    test_marginal_obs_ll_mean, test_marginal_obs_ll_sem = logl_mc_marginal(
        joint_model,
        test_loader,
        data_dim=image_dim,
        mc_sample_size=eval_mc_sample_size,
    )

    train_prior_ll_mean, train_prior_ll_sem = compute_logl(
        joint_model.prior,
        train_loader,
        data_dim=response_dim,
        cond_dim=None,
    )

    val_prior_ll_mean, val_prior_ll_sem = compute_logl(
        joint_model.prior,
        val_loader,
        data_dim=response_dim,
        cond_dim=None,
    )

    test_prior_ll_mean, test_prior_ll_sem = compute_logl(
        joint_model.prior,
        test_loader,
        data_dim=response_dim,
        cond_dim=None,
    )

    return (
        joint_model,
        train_marginal_obs_ll_mean,
        train_marginal_obs_ll_sem,
        val_marginal_obs_ll_mean,
        val_marginal_obs_ll_sem,
        test_marginal_obs_ll_mean,
        test_marginal_obs_ll_sem,
        train_prior_ll_mean,
        train_prior_ll_sem,
        val_prior_ll_mean,
        val_prior_ll_sem,
        test_prior_ll_mean,
        test_prior_ll_sem,
        tracker_output,
        eval_output,
    )
