import torch


def is_pos_def(x):
    return torch.all(torch.linalg.eigvals(x).real > 0)
