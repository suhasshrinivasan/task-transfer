import numpy as np
import torch

marginal_flow_model_config = {
    "dims": None,  # to be determined from data
    "flow_depth": 2,
    "flow_initial_nonlinearity": "inv_softplus",  # flow direction is data -> latent
    "flow_nonlinearity": "tanh",
    "flow_base_distribution": "normal",
}
