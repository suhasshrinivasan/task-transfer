import torch
from torch import nn

from .routines import prepare_init
from .transform_lookup import nonlins, nonneg_transforms


class MLP(nn.Module):
    """
    A general sequential MLP (Multi-Layer Perceptron) module that can be used as a core module before applying readouts.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        n_layers (int): Number of layers in the MLP.
        nonlin (str): Type of nonlinearity to use ('elu', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'none').
        dropout_rate (float): Dropout rate to apply after each layer.
        init_std (float): Standard deviation for weight initialization.
        hidden_features (int): Number of hidden features (calculated as the average of in_features and out_features).
        core_module (nn.Sequential): Sequential container for the MLP layers.
    """

    def __init__(
        self,
        in_features,
        out_features,
        n_layers=1,
        nonlin="relu",
        dropout_rate=0.0,
        init_std=1e-3,
    ):
        """
        Initializes the MLP module with the given parameters.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            n_layers (int): Number of layers in the MLP. Must be >= 1.
            nonlin (str): Type of nonlin to use ('relu', 'tanh', 'sigmoid', 'leaky_relu'). Default is 'relu'.
            dropout_rate (float): Dropout rate to apply after each layer. Default is 0.0.
            init_std (float): Standard deviation for weight initialization. Default is 1e-3.

        Raises:
            ValueError: If n_layers is less than 1.
        """
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        self.hidden_features = (in_features + out_features) // 2
        self.nonlin = nonlin
        # build the sequential MLP module
        self.core_module = nn.Sequential()
        self.core_module.add_module(
            "linear_0", nn.Linear(in_features, self.hidden_features)
        )
        # add dropout after each nn.Linear
        self.core_module.add_module("dropout_0", nn.Dropout(dropout_rate))
        self.core_module.add_module("nonlin_0", nonlins[nonlin]())
        for i in range(1, n_layers):
            self.core_module.add_module(
                f"linear_{i}", nn.Linear(self.hidden_features, self.hidden_features)
            )
            self.core_module.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
            self.core_module.add_module(f"nonlin_{i}", nonlins[nonlin]())

        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        return self.core_module(x)


class LocScaleMLP(nn.Module):
    """
    A PyTorch module for an MLP-based loc-scale model with customizable number
    of layers, nonlin, dropout rate, initialization, and non-negative transformation (for scale).
    This is meant to used to parameterize the loc and scale params of a Normal distribution.
    """

    def __init__(
        self,
        in_features,
        out_features,
        n_layers=2,
        nonlin="relu",
        dropout_rate=0.0,
        init_std=1e-3,
        nonneg_transform="exp",
        clamp_pre_scale=False,
        pre_scale_max=10.0,
    ):
        """
        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            n_layers (int): number of layers in the MLP
            nonlin (str): nonlin to use in the MLP
            dropout_rate (float): dropout rate to use in the MLP
            init_std (float): standard deviation to use for initialization
            nonneg_transform (str): non-negative transform to use for the scale
            clamp_pre_conc (bool): whether to clamp the pre-conc values
            pre_conc_max (float): maximum value for the pre-conc values

        Remarks:
            - The total number of layers in the MLP is n_layers
            - The first n_layers - 1 layers are the core layers
            - The last layer is the loc and pre-scale readout layer
            - The non-negative transform is applied to pre-scale to get scale
            - The pre_scale_max is used to up-clamp the pre-scale values if clamp_pre_scale is True
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        self.nonneg_transform = nonneg_transform
        self.clamp_pre_scale = clamp_pre_scale
        self.pre_scale_max = pre_scale_max

        # build the sequential MLP module
        # subtract 1 layer from n_layers to form n_core_layers
        self.n_core_layers = self.n_layers - 1
        self.core = MLP(
            in_features=in_features,
            out_features=out_features,
            n_layers=self.n_core_layers,
            nonlin=nonlin,
            dropout_rate=dropout_rate,
            init_std=init_std,
        )

        # add the final log conc and log rate readout layers
        # this will make the total number of layers n_layers
        # extract the hidden features from the core module
        self.hidden_features = self.core.hidden_features
        self.loc_module = nn.Linear(self.hidden_features, out_features)
        self.pre_scale_module = nn.Linear(self.hidden_features, out_features)

        # initialize the weights
        self.init()

    def init(self):
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        core_out = self.core(x)

        loc = self.loc_module(core_out)
        pre_scale = self.pre_scale_module(core_out)

        # clamp pre_scale if self.clamp_pre_scale is True
        pre_scale = (
            pre_scale.clamp(min=self.pre_scale_min)
            if self.clamp_pre_scale
            else pre_scale
        )

        # apply the nonzero transforms to get scale
        scale = nonneg_transforms[self.nonneg_transform](pre_scale)

        # ensure scale is always positive
        finfo = torch.finfo(scale.dtype)
        scale = scale.clamp(min=finfo.eps)

        return loc, scale
