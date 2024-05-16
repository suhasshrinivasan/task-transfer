import pickle
from pathlib import Path

import gensn.distributions as G
import torch
from generative_model_configs import gaussian_linear_likelihood, marginal_flow_prior
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from training_configs import generative_model_trainer_config as training_cfg

from experiments.orientation_discrimination.haefner_model import (
    configs as haefner_data_cfg,
)
from task_transfer.evaluation.evaluate_generative_model import visualize_marginal_flow
from task_transfer.ml_lib.model_building import (
    build_conditional,
    build_flow_model,
    build_loc_scale_mlp,
)
from task_transfer.ml_lib.training.loss_criteria import joint_nll
from task_transfer.ml_lib.training.trainer import Trainer
from task_transfer.ml_lib.training.training_tools import TrainLogger

# build dataloaders
# load data
# build model
# train model
# evaluate model
# save model
# save evaluation results
# save configs


def build_dataloaders(data_cfg, training_cfg):
    """
    Build dataloaders for training, validation, and test datasets.

    Args:
        data_cfg (dict): Configuration dictionary for data.
        training_cfg (dict): Configuration dictionary for training.

    Returns:
        tuple: Dataloaders for training, validation, and test datasets.
    """
    with open(data_cfg["data_fname"], "rb") as f:
        data = pickle.load(f)
    train_x = data["x_samples"][
        : int(data_cfg["n_samples"] * training_cfg["train_prop"])
    ]
    train_i = data["i_samples"][
        : int(data_cfg["n_samples"] * training_cfg["train_prop"])
    ]
    val_x = data["x_samples"][
        int(data_cfg["n_samples"] * training_cfg["train_prop"]) : int(
            data_cfg["n_samples"]
            * (training_cfg["train_prop"] + training_cfg["val_prop"])
        )
    ]
    val_i = data["i_samples"][
        int(data_cfg["n_samples"] * training_cfg["train_prop"]) : int(
            data_cfg["n_samples"]
            * (training_cfg["train_prop"] + training_cfg["val_prop"])
        )
    ]
    test_x = data["x_samples"][
        int(
            data_cfg["n_samples"]
            * (training_cfg["train_prop"] + training_cfg["val_prop"])
        ) :
    ]
    test_i = data["i_samples"][
        int(
            data_cfg["n_samples"]
            * (training_cfg["train_prop"] + training_cfg["val_prop"])
        ) :
    ]
    train_loader = DataLoader(
        TensorDataset(train_x, train_i),
        batch_size=training_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_i),
        batch_size=training_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_i),
        batch_size=training_cfg["batch_size"],
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def update_model_cfg(model_cfg, train_loader):
    """
    Update the model configuration based on the training data.

    Args:
        model_cfg (dict): Configuration dictionary for the model.
        train_loader (DataLoader): DataLoader for training data.

    Returns:
        dict: Updated model configuration dictionary.
    """
    responses, images = next(iter(train_loader))
    model_cfg["prior"]["dims"] = responses.shape[1]
    model_cfg["likelihood"]["in_features"] = responses.shape[1]
    model_cfg["likelihood"]["out_features"] = images.shape[1]
    return model_cfg


def build_generative_model(model_cfg):
    """
    Build the generative model using the provided configuration.

    Args:
        model_cfg (dict): Configuration dictionary for the model.

    Returns:
        G.Joint: The generative model.
    """
    prior = build_flow_model(**model_cfg["prior"])
    likelihood = build_loc_scale_mlp(
        in_features=model_cfg["likelihood"]["in_features"],
        out_features=model_cfg["likelihood"]["out_features"],
        n_layers=model_cfg["likelihood"]["n_layers"],
        nonlin=model_cfg["likelihood"]["nonlin"],
        dropout_rate=model_cfg["likelihood"]["dropout_rate"],
        init_std=model_cfg["likelihood"]["init_std"],
        nonneg_transform=model_cfg["likelihood"]["nonneg_transform"],
        clamp_pre_scale=model_cfg["likelihood"]["clamp_pre_scale"],
        pre_scale_max=model_cfg["likelihood"]["pre_scale_max"],
    )
    conditional = build_conditional(
        cond_dist=model_cfg["likelihood"]["cond_dist"],
        likelihood=likelihood,
    )
    conditional = G.IndependentNormal(_parameters=likelihood)
    joint = G.Joint(prior, conditional)
    return joint


def build_trainer(training_cfg, generative_model):
    """
    Build the trainer for the generative model.

    Args:
        training_cfg (dict): Configuration dictionary for training.
        generative_model (G.Joint): The generative model to be trained.

    Returns:
        Trainer: The trainer for the generative model.
    """

    loss_criterion = joint_nll
    eval_criterion = visualize_marginal_flow
    eval_params = {
        "fig_save_dir": Path("/src/project/figures"),
    }
    optimizer = Adam(
        generative_model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    train_logger = TrainLogger(
        model_display_name="GenerativeModel",
        logging_type=training_cfg["logging_type"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        eval_params=eval_params,
        optimizer=optimizer,
        lr=training_cfg["lr"],
        early_stopping_threshold=training_cfg["early_stopping_threshold"],
        early_stopping_patience=training_cfg["early_stopping_patience"],
        logger=train_logger,
        device=device,
    )
    return trainer


def train_generative_model(data_cfg, model_cfg, training_cfg):
    """
    Train the generative model using the provided configurations.

    Args:
        data_cfg (dict): Configuration dictionary for data.
        model_cfg (dict): Configuration dictionary for the model.
        training_cfg (dict): Configuration dictionary for training.

    Returns:
        G.Joint: The trained generative model.
    """
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, training_cfg)
    model_cfg = update_model_cfg(model_cfg, train_loader)
    generative_model = build_generative_model(model_cfg)
    trainer = build_trainer(training_cfg, generative_model)
    model = trainer.train(
        generative_model, train_loader, val_loader, training_cfg["n_epochs"]
    )
    return model


def main():
    """
    Main function to execute the training of the generative model.
    """
    data_cfg = haefner_data_cfg.orginal_haefner_2afc_task1
    model_cfg = {
        "prior": marginal_flow_prior,
        "likelihood": gaussian_linear_likelihood,
    }
    train_generative_model(data_cfg, model_cfg, training_cfg)


if __name__ == "__main__":
    main()
