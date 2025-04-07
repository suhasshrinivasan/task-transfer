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
        optimizer,
        lr,
        early_stopping_threshold,
        early_stopping_patience,
        logger,
        device,
        eval_criterion=None,
        eval_params=None,
        eval_interval=None,
        dj_conn=None,
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
            eval_criterion (callable): Evaluation criterion (function) used for evaluating the model.
            eval_params (dict): Parameters to be passed to the evaluation criterion.
            eval_interval (int): Interval (# epochs) at which to evaluate the model.
        """
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.eval_params = eval_params
        self.eval_interval = eval_interval
        self.optimizer = optimizer
        self.lr = lr
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopper = EarlyStopper(
            patience=early_stopping_patience, min_delta=early_stopping_threshold
        )
        self.logger = logger
        self.device = device
        self.dj_conn = dj_conn

    def train(
        self,
        model,
        train_loader,
        val_loader,
        n_epochs,
        watch_grad_norm=False,
        ping_dj=False,
    ):
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
        # if watch_grad_norm:
        #     torch.autograd.set_detect_anomaly(True)  # TODO: Debugging code. cleanup
        train_losses = []
        val_losses = []
        eval_output = None
        for epoch in range(n_epochs):
            if ping_dj:
                self.dj_conn.ping()
            if (self.eval_criterion is not None) and (epoch % self.eval_interval == 0):
                eval_output = self._eval(model, val_loader, epoch)
            train_loss = self._train(model, train_loader, epoch, watch_grad_norm)
            train_losses.append(train_loss)
            val_loss = self._val(model, val_loader, epoch)
            val_losses.append(val_loss)
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
                if self.eval_criterion is not None:
                    eval_output = self._eval(model, val_loader, epoch)
                return {
                    "tracker_output": {
                        "train_loss": train_losses,
                        "val_loss": val_losses,
                    },
                    "eval_output": eval_output,
                }

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
        if self.eval_criterion is not None:
            eval_output = self._eval(model, val_loader, epoch)
        return {
            "tracker_output": {"train_loss": train_losses, "val_loss": val_losses},
            "eval_output": eval_output,
        }

    def _train(self, model, train_loader, epoch, watch_grad_norm=False):
        """
        Performs one training epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
            watch_grad_norm (bool): Whether to log the gradient norm.

        Returns:
            float: Mean training loss for the epoch.
        """
        model = model.to(self.device)
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch = [x.to(self.device) for x in batch]
            loss = self.loss_criterion(model, batch).mean()
            loss.backward()
            # TODO: watch grad norm code is commented out. Uncomment if necessary
            # if watch_grad_norm:
            #     grad_norm = torch.norm(
            #         torch.cat([p.grad.flatten() for p in model.parameters()])
            #     ).item()
            #     self.logger.log(
            #         {"grad_norm": grad_norm},
            #         f"Train Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{len(train_loader)}",
            #     )
            train_losses.append(loss.item())
            self.optimizer.step()
            # TODO: log metrics every batch if necessary otherwise it'll cause a significant slowdown
            # metrics = {"train_batch_loss": loss.item()}
            # self.logger.log(
            #     metrics, f"Train Epoch: {epoch + 1} {batch_idx + 1}/{len(train_loader)}"
            # )
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
                batch = [x.to(self.device) for x in batch]
                loss = self.loss_criterion(model, batch).mean()
                val_losses.append(loss.item())
                # TODO: log metrics every batch if necessary otherwise it'll cause a significant slowdown
                # metrics = {"val_batch_loss": loss.item()}
                # self.logger.log(
                #     metrics, f"Val Epoch: {epoch + 1} {batch_idx + 1}/{len(val_loader)}"
                # )
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
        with torch.no_grad():
            model.eval()
            return self.eval_criterion(
                model, val_loader, epoch, self.device, self.eval_params, self.logger
            )
