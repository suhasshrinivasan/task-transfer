marginal_flow_prior = {
    "dims": None,  # to be determined from data
    "flow_depth": 2,
    "flow_initial_nonlinearity": "inv_softplus",  # flow direction is data -> latent
    "flow_nonlinearity": "tanh",
    "flow_base_distribution": "normal",
}

gaussian_linear_likelihood = {
    "cond_dist": "indep_normal",
    "nonneg_transform": "exp",
    "in_features": None,  # to be determined from data
    "out_features": None,  # to be determined from data
    "n_layers": 1,
    "nonlin": "none",
    "dropout_rate": 0.0,
    "init_std": 1e-3,
    "clamp_pre_scale": True,
    "pre_scale_max": 10.0,
}
