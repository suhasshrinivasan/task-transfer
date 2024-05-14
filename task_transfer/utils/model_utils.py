import torch

from ..sampling_models.sampling_models import HaefnerModel
from .insilico_stimuli import generate_gabors


def build_haefner_model(
    p_c, c1_psi, c2_psi, kappa, g_phi, delta, lam, x_phi, obs_sigma
):
    """
    Builds and returns an instance of the HaefnerModel.

    Args:
        p_c (float): Probability of the binary latent variable `c` being 1.
        c1_psi (float): Orientation value for `c` being 0.
        c2_psi (float): Orientation value for `c` being 1.
        kappa (float): Concentration parameter for the von Mises distribution in g population.
        g_phi (torch.Tensor): Tuning curves for the g population.
        delta (float): Scaling factor for the feedback tuning.
        lam (float): Concentration parameter for the von Mises distribution in x feedback tuning.
        x_phi (torch.Tensor): Tuning curves for the x population.
        obs_sigma (float): Standard deviation of the observational noise in the I samples.

    Returns:
        HaefnerModel: An instance of the HaefnerModel initialized with the provided parameters.
    """
    x_pfs = torch.Tensor(generate_gabors(orientations=x_phi.tolist()))
    return HaefnerModel(
        p_c=p_c,
        c1_psi=c1_psi,
        c2_psi=c2_psi,
        kappa=kappa,
        g_phi=g_phi,
        delta=delta,
        lam=lam,
        x_phi=x_phi,
        x_pfs=x_pfs,
        obs_sigma=obs_sigma,
    )
