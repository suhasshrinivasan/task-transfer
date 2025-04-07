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
            n_layers (int): Number of layers in the MLP. If 0, the core module is set to nn.Identity(). Default is 1.
            nonlin (str): Type of nonlin to use ('relu', 'tanh', 'sigmoid', 'leaky_relu'). Default is 'relu'.
            dropout_rate (float): Dropout rate to apply after each layer. Default is 0.0.
            init_std (float): Standard deviation for weight initialization. Default is 1e-3.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.init_std = init_std
        self.nonlin = nonlin
        if n_layers == 0:
            self.core_module = nn.Identity()
        else:
            # build the sequential MLP module
            self.core_module = nn.Sequential()
            if n_layers == 1:
                self.hidden_features = out_features
            else:
                self.hidden_features = out_features
            self.core_module.add_module(
                "linear_0", nn.Linear(in_features, self.hidden_features)
            )
            # add dropout after each nn.Linear
            self.core_module.add_module("dropout_0", nn.Dropout(dropout_rate))
            self.core_module.add_module("nonlin_0", nonlins[nonlin]())
            for i in range(1, n_layers - 1):
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


class LocScale(nn.Module):
    """
    A module that predicts location (mean) and scale (standard deviation) parameters
    for a distribution based on the output of a core neural network.

    Attributes:
        core_nn (nn.Module): The core neural network whose output is used to predict the location and scale.
        init_std (float): Standard deviation for initializing the weights. Default is 1e-3.
        nonneg_transform (str): Transformation to ensure scale is non-negative. Default is 'exp'.
        clamp_pre_scale (bool): Whether to clamp the pre-scale values. Default is False.
        pre_scale_max (float): Maximum value for clamping pre-scale values. Default is 10.0.
        loc_module (nn.Linear): Linear layer to predict the location parameter.
        pre_scale_module (nn.Linear): Linear layer to predict the pre-scale parameter.
    """

    def __init__(
        self,
        core_nn,
        out_features_loc,
        out_features_scale,
        init_std=1e-3,
        nonneg_transform="exp",
        clamp_pre_scale=False,
        pre_scale_max=10.0,
    ):
        """
        Initializes the LocScale module with the given parameters.

        Args:
            core_nn (nn.Module): The core neural network whose output is used to predict the location and scale.
            init_std (float): Standard deviation for initializing the weights. Default is 1e-3.
            nonneg_transform (str): Transformation to ensure scale is non-negative. Default is 'exp'.
            clamp_pre_scale (bool): Whether to clamp the pre-scale values. Default is False.
            pre_scale_max (float): Maximum value for clamping pre-scale values. Default is 10.0.
        """
        super().__init__()
        self.core_nn = core_nn
        self.out_features_loc = out_features_loc
        self.out_features_scale = out_features_scale
        self.init_std = init_std
        self.nonneg_transform = nonneg_transform
        self.clamp_pre_scale = clamp_pre_scale
        self.pre_scale_max = pre_scale_max

        self.loc_module = nn.Linear(self.core_nn.out_features, self.out_features_loc)
        self.pre_scale_module = nn.Linear(
            self.core_nn.out_features, self.out_features_scale
        )

        self.init()

    def init(self):
        """
        Initializes the weights of the LocScale module using a normal distribution for weights
        and a constant value for biases.
        """
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        """
        Defines the forward pass through the LocScale module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing:
                - loc (torch.Tensor): The predicted location (mean) parameter.
                - scale (torch.Tensor): The predicted scale (standard deviation) parameter.
        """
        core_out = self.core_nn(x)

        loc = self.loc_module(core_out)
        pre_scale = self.pre_scale_module(core_out)

        # clamp pre_scale if self.clamp_pre_scale is True
        pre_scale = (
            pre_scale.clamp(max=self.pre_scale_max)
            if self.clamp_pre_scale
            else pre_scale
        )

        # apply the nonzero transforms to get scale
        scale = nonneg_transforms[self.nonneg_transform](pre_scale)

        # ensure scale is always positive
        finfo = torch.finfo(scale.dtype)
        scale = scale.clamp(min=finfo.eps)

        return loc, scale


class ConcRate(nn.Module):
    """
    A module that predicts conc and rate parameters for a Gamma distribution
    based on the output of a core neural network.

    Attributes:
        core_nn (nn.Module): The core neural network whose output is used to predict the conc and rate.
        init_std (float): Standard deviation for initializing the weights. Default is 1e-3.
        nonneg_transform (str): Transformation to ensure conc and rate are non-negative. Default is 'exp'.
        clamp_pre_conc (bool): Whether to clamp the pre-conc values. Default is False.
        pre_conc_max (float): Maximum value for clamping pre-conc values. Default is 4.0.
        clamp_pre_rate (bool): Whether to clamp the pre-rate values. Default is False.
        pre_rate_min (float): Minimum value for clamping pre-rate values. Default is -1.6.
        conc_module (nn.Linear): Linear layer to predict the conc parameter.
        rate_module (nn.Linear): Linear layer to predict the rate parameter.
    """

    def __init__(
        self,
        core_nn,
        out_features_conc,
        out_features_rate,
        init_std=1e-3,
        nonneg_transform="exp",
        clamp_pre_conc=False,
        pre_conc_max=4.0,
        clamp_pre_rate=False,
        pre_rate_min=-1.6,
    ):
        """
        Initializes the ConcRate module with the given parameters.

        Args:
            core_nn (nn.Module): The core neural network whose output is used to predict the conc and rate.
            out_features_conc (int): Number of output features for the conc parameter.
            out_features_rate (int): Number of output features for the rate parameter.
            init_std (float): Standard deviation for initializing the weights. Default is 1e-3.
            nonneg_transform (str): Transformation to ensure conc and rate are non-negative. Default is 'exp'.
            clamp_pre_conc (bool): Whether to clamp the pre-conc values. Default is False.
            pre_conc_max (float): Maximum value for clamping pre-conc values. Default is 4.0.
            clamp_pre_rate (bool): Whether to clamp the pre-rate values. Default is False.
            pre_rate_min (float): Minimum value for clamping pre-rate values. Default is -1.6.

        Remarks:
            - The default value for pre_conc_max is 4.0, chosen such that the exp transform
                on pre_conc_max is exp(4.0) = 54.6, which is close to the maximum value of
                neuronal responses observed in the dataset as this module is generally used
                for modeling response distribution
            - Similarly, we also do an up-clamp on the pre-rate values such that the rate
                values are >= 0.2 (which would already be quite a wide gamma distribution), hence
                default value of pre_rate_min is -1.6 (log(0.2) = -1.6)
        """
        super().__init__()
        self.core_nn = core_nn
        self.out_features_conc = out_features_conc
        self.out_features_rate = out_features_rate
        self.init_std = init_std
        self.nonneg_transform = nonneg_transform
        self.clamp_pre_conc = torch.tensor(clamp_pre_conc, dtype=torch.bool)
        self.pre_conc_max = pre_conc_max
        self.clamp_pre_rate = torch.tensor(clamp_pre_rate, dtype=torch.bool)
        self.pre_rate_min = pre_rate_min

        self.conc_module = nn.Linear(self.core_nn.out_features, self.out_features_conc)
        self.rate_module = nn.Linear(self.core_nn.out_features, self.out_features_rate)

        self.init()

    def init(self):
        """
        Initializes the weights of the ConcRate module using a normal distribution for weights
        and a constant value for biases.
        """
        self.apply(prepare_init(self.init_std))

    def forward(self, x):
        """
        Defines the forward pass through the ConcRate module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing:
                - conc (torch.Tensor): The predicted conc parameter.
            - rate (torch.Tensor): The predicted rate parameter.
        """
        core_out = self.core_nn(x)

        pre_conc = self.conc_module(core_out)
        pre_rate = self.rate_module(core_out)

        # Apply clamping if needed using torch.where
        pre_conc = torch.where(
            self.clamp_pre_conc, pre_conc.clamp(max=self.pre_conc_max), pre_conc
        )
        pre_rate = torch.where(
            self.clamp_pre_rate, pre_rate.clamp(min=self.pre_rate_min), pre_rate
        )

        # Apply the nonnegative transforms to get conc and rate
        conc = nonneg_transforms[self.nonneg_transform](pre_conc)
        rate = nonneg_transforms[self.nonneg_transform](pre_rate)

        # Ensure conc and rate are always positive
        finfo = torch.finfo(rate.dtype)
        conc = conc.clamp(min=finfo.eps)
        rate = rate.clamp(min=finfo.eps)

        return conc, rate
