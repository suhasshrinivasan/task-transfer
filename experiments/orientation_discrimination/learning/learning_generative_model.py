import pickle

import gensn.distributions as G
import torch
from haefner_model import configs as cfg
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from task_transfer.ml_lib.model_building import build_flow_model
from task_transfer.ml_lib.modules import LocScaleMLP
from task_transfer.ml_lib.training.loss_criteria import joint_nll
from task_transfer.ml_lib.training.trainer import Trainer
from task_transfer.ml_lib.training.training_tools import TrainLogger

from .generative_model_configs import gaussian_linear_likelihood, marginal_flow_prior

task_cfg = cfg.orginal_haefner_2afc_task1
data_fname = task_cfg["data_fname"]
with open(data_fname, "rb") as f:
    data = pickle.load(f)

data["x_samples"].shape, data["i_samples"].shape

datadir = cfg.orginal_haefner_2afc_task1.data_dir
task_dataset = cfg


# build dataloaders
# load data
# build model
# train model
# evaluate model
# save model
# save evaluation results
# save configs


def build_dataloaders(data_cfg, training_cfg):
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
    responses, images = next(iter(train_loader))
    model_cfg["prior"]["dims"] = responses.shape[1]
    model_cfg["likelihood"]["in_features"] = images.shape[1]
    model_cfg["likelihood"]["out_features"] = responses.shape[1]
    return model_cfg


def build_generative_model(model_cfg):
    prior = build_flow_model(**model_cfg["prior"])
    likelihood = LocScaleMLP(**model_cfg["likelihood"])
    conditional = G.IndependentNormal(_parameters=likelihood)
    joint = G.Joint(prior, conditional)
    return joint


def build_trainer(training_cfg, generative_model):
    loss_criterion = joint_nll
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
        optimizer=optimizer,
        lr=training_cfg["lr"],
        early_stopping_threshold=training_cfg["early_stopping_threshold"],
        early_stopping_patience=training_cfg["early_stopping_patience"],
        logger=train_logger,
        device=device,
    )
    return trainer


def train_generative_model(data_cfg, model_cfg, training_cfg):
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, training_cfg)
    model_cfg = update_model_cfg(model_cfg, train_loader)
    generative_model = build_generative_model(model_cfg)
    trainer = build_trainer(training_cfg, generative_model)
    model = trainer.train(generative_model, train_loader, val_loader)
    return model
