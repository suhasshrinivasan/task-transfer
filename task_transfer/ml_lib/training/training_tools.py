import numpy as np
import wandb

from ..routines import copy_model_state


class EarlyStopper:
    """
    Early stopping utility to monitor validation loss and stop training when it stops improving.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        counter (int): Number of epochs since the last improvement.
        best_model_state_dict (dict): State dictionary of the best model.
        min_validation_loss (float): The lowest validation loss observed.

    Methods:
        step(validation_loss, model): Updates the early stopping counter and checks if training should stop.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Initializes the EarlyStopper with the given parameters.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped. Default is 1.
            min_delta (float): Minimum change in the monitored value to qualify as an improvement. Default is 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_model_state_dict = None
        self.min_validation_loss = np.inf

    def step(self, validation_loss, model):
        """
        Updates the early stopping counter and checks if training should stop.

        Args:
            validation_loss (float): The validation loss for the current epoch.
            model (torch.nn.Module): The model being trained.

        Returns:
            dict or None: The state dictionary of the best model if early stopping is triggered, otherwise None.
        """
        returnable_state_dict = None
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            print(f"New best validation loss: {self.min_validation_loss}")
            self.best_model_state_dict = copy_model_state(model)
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                returnable_state_dict = self.best_model_state_dict
        return returnable_state_dict

    def __repr__(self):
        """
        Returns a string representation of the EarlyStopper instance.

        Returns:
            str: String representation of the EarlyStopper instance.
        """
        return f"EarlyStopper(patience={self.patience}, min_delta={self.min_delta})"


class TrainLogger:
    """
    Logger utility to record training and validation metrics.

    Attributes:
        model_display_name (str): Display name for the model in the logs.
        logging_type (str): Type of logging to use ('wandb' or 'stdout').

    Methods:
        log(metrics, count_phrase): Logs the provided metrics.
    """

    def __init__(self, model_display_name, logging_type="wandb"):
        """
        Initializes the TrainLogger with the given parameters.

        Args:
            model_display_name (str): Display name for the model in the logs.
            logging_type (str): Type of logging to use ('wandb' or 'stdout'). Default is 'wandb'.
        """
        self.model_display_name = model_display_name
        self.logging_type = logging_type

    def __repr__(self):
        """
        Returns a string representation of the TrainLogger instance.

        Returns:
            str: String representation of the TrainLogger instance.
        """
        return f"TrainLogger(model_display_name={self.model_display_name}, logging_type={self.logging_type})"

    def log(self, metrics, count_phrase):
        """
        Logs the provided metrics.

        Args:
            metrics (dict): Dictionary of metrics to log.
            count_phrase (str): A phrase indicating the current epoch and batch count.

        Raises:
            ValueError: If the logging_type is not one of 'wandb' or 'stdout'.
        """
        # Append model_display_name to all keys
        metrics = {f"{self.model_display_name}/{k}": v for k, v in metrics.items()}
        if self.logging_type == "wandb":
            wandb.log(metrics)
        elif self.logging_type == "stdout":
            print(f"{count_phrase}:\n {metrics}")
        else:
            raise ValueError("logging_type must be one of 'wandb' or 'stdout'")
