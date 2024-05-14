import torch
import torch.distributions as D

from ..utils.math_utils import cos2_von_mises


class HaefnerModel:
    """
    A class for the Haefner model, a probabilistic model used for sampling and
    estimating the posterior distribution of certain variables given observed data.

    Attributes:
        p_c (float): Probability of the binary latent variable `c` being 1.
        c1_psi (float): Orientation value used when the binary variable `c` is 0.
        c2_psi (float): Orientation value used when the binary variable `c` is 1.
        kappa (float): Concentration parameter for the von Mises distribution in the g population.
        g_phi (torch.Tensor): Preferred orientations for the g population.
        delta (float): Feedback weight for the x population.
        lam (float): Concentration parameter for the von Mises distribution in the x feedback tuning.
        x_phi (torch.Tensor): Preferred orientations the x population.
        x_pfs (torch.Tensor): Projection fields for the x population.
        obs_sigma (float): Standard deviation of the observational noise in the I samples.
        n_x (int): number of x population.
        n_g (int): number of g population.
    """

    def __init__(
        self,
        p_c,
        c1_psi,
        c2_psi,
        kappa,
        g_phi,
        delta,
        lam,
        x_phi,
        x_pfs,
        obs_sigma,
    ):
        """
        Initializes the HaefnerModel with specified parameters.

        Args:
            p_c (float): Probability of the binary latent variable `c` being 1.
            c1_psi (float): Orientation value used when the binary variable `c` is 0.
            c2_psi (float): Orientation value used when the binary variable `c` is 1.
            kappa (float): Concentration parameter for the von Mises distribution in the g population.
            g_phi (torch.Tensor): Preferred orientations for the g population.
            delta (float): Feedback weight for the x population.
            lam (float): Concentration parameter for the von Mises distribution in the x feedback tuning.
            x_phi (torch.Tensor): Preferred orientations the x population.
            x_pfs (torch.Tensor): Projection fields for the x population.
            obs_sigma (float): Standard deviation of the observational noise in the I samples.
        """
        self.p_c = p_c
        self.c1_psi = c1_psi
        self.c2_psi = c2_psi
        self.g_phi = g_phi
        self.n_g = len(g_phi)
        self.kappa = kappa
        self.delta = delta
        self.lam = lam
        self.x_phi = x_phi
        self.n_x = len(x_phi)
        self.x_pfs = x_pfs
        self.obs_sigma = obs_sigma

    def g_tuning(self, o):
        """
        Computes the tuning probabilities for the g population.

        Args:
            o (torch.Tensor): Difference between orientation samples and `g_phi`.

        Returns:
            torch.Tensor: Tuning probabilities for the g population.
        """
        return (
            2
            * torch.pi
            / self.n_g
            * cos2_von_mises(x=o, loc=0.0, concentration=self.kappa)
        )

    def x_feedback_tuning(self, o):
        """
        Computes the feedback tuning probabilities for the x population.

        Args:
            o (torch.Tensor): Difference between the g and x populations.

        Returns:
            torch.Tensor: Feedback tuning probabilities for the x population.
        """
        return cos2_von_mises(
            o,
            loc=0.0,
            concentration=self.lam,
        )

    def x_tuning(self, x_fb):
        """
        Adjusts the feedback tuning values for the x population with a scaling factor.

        Args:
            x_fb (torch.Tensor): Feedback tuning values for the x population.

        Returns:
            torch.Tensor: Adjusted feedback tuning values.
        """
        return 1 + self.delta * x_fb

    def sample_prior(self, n_samples=1):
        """
        Samples from the prior distribution of the model.

        Args:
            n_samples (int): Number of samples to generate. Default is 1.

        Returns:
            dict: A dictionary containing sampled values:
                - "c_samples" (torch.Tensor): Samples of the binary variable `c`.
                - "g_prob" (torch.Tensor): Tuning probabilities for the g population.
                - "g_samples" (torch.Tensor): Samples for the g population.
                - "x_fb" (torch.Tensor): Feedback tuning for the x population.
                - "tau" (torch.Tensor): Adjusted feedback tuning values.
                - "x_samples" (torch.Tensor): Samples for the x population.
                - "i_samples" (torch.Tensor): Observed samples with added noise.
        """
        c_samples = D.Bernoulli(probs=torch.tensor([self.p_c])).sample((n_samples,))
        orientation_samples = torch.where(c_samples == 0.0, self.c1_psi, self.c2_psi)
        diff_g_orientations = orientation_samples - self.g_phi.reshape(1, -1)
        g_probs = self.g_tuning(diff_g_orientations)
        g_samples = D.Bernoulli(probs=g_probs).sample()
        x_fb = g_samples @ self.x_feedback_tuning(
            self.g_phi.unsqueeze(-1) - self.x_phi.unsqueeze(0)
        )
        tau = self.x_tuning(x_fb)
        x_samples = D.Exponential(rate=1 / tau).sample()
        i_loc = x_samples @ self.x_pfs.reshape(self.x_pfs.shape[0], -1)
        i_samples = D.Normal(loc=i_loc, scale=self.obs_sigma).sample()
        samples_dict = {
            "c_samples": c_samples,
            "g_prob": g_probs,
            "g_samples": g_samples,
            "x_fb": x_fb,
            "tau": tau,
            "x_samples": x_samples,
            "i_samples": i_samples,
        }
        return samples_dict
