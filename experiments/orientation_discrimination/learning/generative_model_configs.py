marginal_flow_prior = {
    "dims": None,  # to be determined from data
    "flow_depth": 2,
    "flow_initial_nonlinearity": "inv_softplus",  # flow direction is data -> latent
    "flow_nonlinearity": "tanh",
    "flow_base_distribution": "normal",
}

gaussian_linear_likelihood = {
    "in_features": None,  # to be determined from data
    "out_features": None,  # to be determined from data
    "n_layers": 1,
    "nonlin": "none",
    "dropout_rate": 0.0,
    "init_std": 1e-3,
    "pre_clamp_scale": True,
}
