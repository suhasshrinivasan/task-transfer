import pickle

from torch.utils.data import DataLoader, TensorDataset


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
