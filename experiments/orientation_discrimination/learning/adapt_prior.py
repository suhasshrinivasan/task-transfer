import pickle
from pathlib import Path

import gensn.distributions as G
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from generative_model_configs import (
    full_flow_prior,
    gaussian_linear_likelihood,
    marginal_flow_prior,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from training_configs import (
    generative_model_trainer_config,
    generative_model_trainer_config2,
)

from experiments.orientation_discrimination.haefner_model import (
    configs as haefner_data_cfg,
)
from task_transfer.ml_lib.model_building import (
    build_conditional,
    build_flow_model,
    build_loc_scale_mlp,
)
from task_transfer.ml_lib.training.loss_criteria import joint_nll
from task_transfer.ml_lib.training.trainer import Trainer
from task_transfer.ml_lib.training.training_tools import TrainLogger


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


def main():
    """
    Main function to execute the training of the generative model.
    """
    data_cfg = haefner_data_cfg.orginal_haefner_2afc_task2
    # model_cfg = {
    #     "prior": marginal_flow_prior,
    #     "likelihood": gaussian_linear_likelihood,
    # }
    model_cfg = {
        "prior": full_flow_prior,
        "likelihood": gaussian_linear_likelihood,
    }
    # train_generative_model(data_cfg, model_cfg, training_cfg)
    train_generative_model(data_cfg, model_cfg, generative_model_trainer_config2)


if __name__ == "__main__":
    main()
