import tempfile

import torch

from task_transfer.evaluation.evaluate_generative_model import logl_conditional
from task_transfer.ml_lib.data_loading import build_dataloaders


def evaluate_predictive_model(
    model_args,
    dataloader_args,
    device,
):
    with tempfile.TemporaryFile() as tempdir:
        model = torch.load(model_args["model"])
        train_loader, val_loader, test_loader = build_dataloaders(
            dataloader_args["data_fname"],
            dataloader_args["train_prop"],
            dataloader_args["val_prop"],
            dataloader_args["batch_size"],
        )
        response_dim = 0
        image_dim = 1
        train_ll_mean, train_ll_sem = logl_conditional(
            model=model,
            data_loader=train_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=device,
        )
        val_ll_mean, val_ll_sem = logl_conditional(
            model=model,
            data_loader=val_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=device,
        )
        test_ll_mean, test_ll_sem = logl_conditional(
            model=model,
            data_loader=test_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=device,
        )
    return (
        train_ll_mean,
        train_ll_sem,
        val_ll_mean,
        val_ll_sem,
        test_ll_mean,
        test_ll_sem,
    )
