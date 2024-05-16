import numpy as np
import torch

from .training_tools import EarlyStopper


class Trainer:
    """
    Class for training a PyTorch model with early stopping.

    Attributes:
        optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        lr (float): Learning rate for the optimizer.
        early_stopping_threshold (float): Minimum change in the monitored value to qualify as an improvement.
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
        early_stopper (EarlyStopper): Instance of the EarlyStopper class to handle early stopping logic.
        logger (Logger): Logger instance to record training and validation metrics.
        device (torch.device): Device on which to train the model (CPU or GPU).
    """

    def __init__(
        self,
        loss_criterion,
        eval_criterion,
        eval_params,
        optimizer,
        lr,
        early_stopping_threshold,
        early_stopping_patience,
        logger,
        device,
    ):
        """
        Initializes the Trainer class with the given parameters.

        Args:
            loss_criterion (callable): Loss criterion (function) used for training the model.
            optimizer (torch.optim.Optimizer): Optimizer used for training the model.
            lr (float): Learning rate for the optimizer.
            early_stopping_threshold (float): Minimum change in the monitored value to qualify as an improvement.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            logger (Logger): Logger instance to record training and validation metrics.
            device (torch.device): Device on which to train the model (CPU or GPU).
        """
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.eval_params = eval_params
        self.optimizer = optimizer
        self.lr = lr
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopper = EarlyStopper(
            patience=early_stopping_patience, min_delta=early_stopping_threshold
        )
        self.logger = logger
        self.device = device

    def train(self, model, train_loader, val_loader, n_epochs):
        """
        Trains the model and applies early stopping if validation loss does not improve.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            n_epochs (int): Number of epochs to train the model.

        Returns:
            torch.nn.Module: The best model based on validation loss.
        """
        for epoch in range(n_epochs):
            train_loss = self._train(model, train_loader, epoch)
            val_loss = self._val(model, val_loader, epoch)
            track_message = f"Epoch {epoch + 1}/{n_epochs}"

            # Early stopping check
            if best_state_dict := self.early_stopper.step(val_loss, model):
                print("Early stopping triggered")
                model.load_state_dict(best_state_dict)
                best_val_loss = self._val(model, val_loader, epoch)
                metrics = {
                    "val_loss": best_val_loss,
                }
                self.logger.log(metrics, track_message)
                self._eval(model, val_loader, epoch)
                return model

            if epoch % 10 == 0:
                self._eval(model, val_loader, epoch)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.logger.log(metrics, track_message)

        print("Training complete. Fetching best model.")
        best_state_dict = self.early_stopper.best_model_state_dict
        model.load_state_dict(best_state_dict)
        best_val_loss = self._val(model, val_loader, epoch)
        metrics = {
            "val_loss": best_val_loss,
        }
        self.logger.log(metrics, track_message)
        return model

    def _train(self, model, train_loader, epoch):
        """
        Performs one training epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.

        Returns:
            float: Mean training loss for the epoch.
        """
        model = model.to(self.device)
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.loss_criterion(model, batch).mean()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch + 1} {batch_idx + 1}/{len(train_loader)}")
        return np.mean(train_losses)

    def _val(self, model, val_loader, epoch):
        """
        Evaluates the model on the validation dataset.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.

        Returns:
            float: Mean validation loss for the epoch.
        """
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss = self.loss_criterion(model, batch).mean()
                val_losses.append(loss.item())
                if batch_idx % 10 == 0:
                    print(f"Val Epoch: {epoch + 1} {batch_idx + 1}/{len(val_loader)}")
        return np.mean(val_losses)

    def _eval(self, model, val_loader, epoch):
        """
        Evaluates the model on the validation dataset using custom evaluation logic

        Args:
            model (torch.nn.Module): The model to be evaluated.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        """
        self.eval_criterion(
            model.prior,
            val_loader,
            epoch,
            **self.eval_params,
        )
