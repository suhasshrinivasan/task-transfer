import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_dataloaders(data_fname, train_prop, val_prop, batch_size):
    """
    Build dataloaders for training, validation, and test datasets.

    Args:
        data_cfg (dict): Configuration dictionary for data.
        training_cfg (dict): Configuration dictionary for training.

    Returns:
        tuple: Dataloaders for training, validation, and test datasets.
    """
    with open(data_fname, "rb") as f:
        data = pickle.load(f)
    n_samples = data["x_samples"].shape[0]
    n_train_samples = int(n_samples * train_prop)
    n_val_samples = int(n_samples * val_prop)
    train_x = data["x_samples"][:n_train_samples]
    train_i = data["i_samples"][:n_train_samples]
    val_x = data["x_samples"][n_train_samples : (n_train_samples + n_val_samples)]
    val_i = data["i_samples"][n_train_samples : (n_train_samples + n_val_samples)]
    test_x = data["x_samples"][(n_train_samples + n_val_samples) :]
    test_i = data["i_samples"][(n_train_samples + n_val_samples) :]

    train_loader = DataLoader(
        TensorDataset(train_x, train_i), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_i), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_i), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


def build_dataloaders_from_samples_paths(
    response_samples_path, obs_samples_path, train_prop, val_prop, batch_size, seed
):
    """
    Build dataloaders for training, validation, and test datasets.

    Args:
        response_samples_path (str): Path to response samples.
        obs_samples_path (str): Path to observed samples.
        train_prop (float): Proportion of samples to use for training.
        val_prop (float): Proportion of samples to use for validation.
        batch_size (int): Batch size for dataloaders.

    Returns:
        tuple: Dataloaders for training, validation, and test datasets.
    """
    # Load response and observed samples
    response_samples = torch.load(response_samples_path)
    obs_samples = torch.load(obs_samples_path)

    return build_dataloaders_from_samples(
        response_samples, obs_samples, train_prop, val_prop, batch_size, seed
    )


def build_dataloaders_from_samples(
    response_samples, obs_samples, train_prop, val_prop, batch_size, seed
):
    """
    Build dataloaders for training, validation, and test datasets.

    Args:
        response_samples (torch.Tensor): Response samples.
        obs_samples (torch.Tensor): Observed samples.
        train_prop (float): Proportion of samples to use for training.
        val_prop (float): Proportion of samples to use for validation.
        batch_size (int): Batch size for dataloaders.

    Returns:
        tuple: Dataloaders for training, validation, and test datasets.
    """
    # Set the random seed
    torch.manual_seed(seed)
    # Ensure the proportions sum to less than or equal to 1
    assert (
        train_prop + val_prop <= 1.0
    ), "Training and validation proportions must sum to less than or equal to 1"

    # Combine response and observed samples into a single dataset
    dataset = TensorDataset(response_samples, obs_samples)

    # Calculate the number of samples for training, validation, and test
    total_samples = len(dataset)
    train_size = int(train_prop * total_samples)
    val_size = int(val_prop * total_samples)
    test_size = total_samples - train_size - val_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
