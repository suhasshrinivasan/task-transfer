import torch
from scipy.special import i0


def is_pos_def(x):
    return torch.all(torch.linalg.eigvals(x).real > 0)


def cos2_von_mises(x, loc, concentration, normalized=True):
    exponent = torch.exp(concentration * torch.cos(2 * (x - loc)))
    if normalized:
        return exponent / (2 * torch.pi * i0(concentration))
    else:
        return exponent
