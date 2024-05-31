import torch
from gensn.transforms import invertible as inv_nn
from torch import nn

inv_nonlins = {
    "elu": inv_nn.ELU,
    "inv_elu": lambda: inv_nn.InverseTransform(inv_nn.ELU()),
    "tanh": inv_nn.Tanh,
    "inv_tanh": lambda: inv_nn.InverseTransform(inv_nn.Tanh()),
    "sigmoid": inv_nn.Sigmoid,
    "inv_sigmoid": lambda: inv_nn.InverseTransform(inv_nn.Sigmoid()),
    "exp": inv_nn.Exp,
    "inv_exp": lambda: inv_nn.InverseTransform(inv_nn.Exp()),
    "softplus": inv_nn.Softplus,
    "inv_softplus": lambda: inv_nn.InverseTransform(inv_nn.Softplus()),
    "eluplus1": inv_nn.ELUplus1,
    "inv_eluplus1": lambda: inv_nn.InverseTransform(inv_nn.ELUplus1()),
    "log": inv_nn.Log,
    "inv_log": lambda: inv_nn.InverseTransform(inv_nn.Log()),
    "pow2": lambda: inv_nn.Pow(2),
    "inv_pow2": lambda: inv_nn.InverseTransform(inv_nn.Pow(2)),
    "pow3": lambda: inv_nn.Pow(3),
    "inv_pow3": lambda: inv_nn.InverseTransform(inv_nn.Pow(3)),
    "sqrt": inv_nn.Sqrt,
    "inv_sqrt": lambda: inv_nn.InverseTransform(inv_nn.Sqrt()),
    "leaky_relu": inv_nn.LeakyReLU,
    "inv_leaky_relu": lambda: inv_nn.InverseTransform(inv_nn.LeakyReLU()),
}

nonlins = {
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}

nonneg_transforms = {
    "exp": torch.exp,
    "softplus": nn.functional.softplus,
    "relu": torch.relu,
    "eluplus1": lambda x: nn.functional.elu(x) + 1,
    "abs": torch.abs,
    "sq": lambda x: x**2,
    "sigmoid": torch.sigmoid,
    "none": lambda x: x,
}
