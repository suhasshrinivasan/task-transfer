import torch
import torch.distributions as D

from ..utils.insilico_stimuli import generate_gabors
from ..utils.math_utils import cos2_von_mises


class OrientationDiscriminationModel:
    def __init__(
        self,
        c1_prob,
        c1_loc,
        c1_std,
        c2_loc,
        c2_std,
        g_pref_orientations,
        g_tuning_specificity,
        x_feedback_weight,
        x_tuning_specificity,
        x_pref_orientations,
        obs_noise,
    ):
        self.c1_prob = c1_prob
        self.c1_loc = c1_loc
        self.c1_std = c1_std
        self.c2_loc = c2_loc
        self.c2_std = c2_std
        self.g_pref_orientations = g_pref_orientations
        self.n_g = len(g_pref_orientations)
        self.g_tuning_specificity = g_tuning_specificity
        self.x_feedback_weight = x_feedback_weight
        self.x_tuning_specificity = x_tuning_specificity
        self.x_pref_orientations = x_pref_orientations
        self.n_x = len(x_pref_orientations)
        self.x_pfs = torch.Tensor(
            generate_gabors(orientations=x_pref_orientations.tolist())
        )
        self.obs_noise = obs_noise

    def g_tuning(self, o):
        return (
            2
            * torch.pi
            / self.n_g
            * cos2_von_mises(x=o, loc=0.0, concentration=self.g_tuning_specificity)
        )

    def x_feedback_tuning(self, o):
        return cos2_von_mises(
            o,
            loc=0.0,
            concentration=self.x_tuning_specificity,
        )

    def x_tuning(self, x_feedback_factor):
        return 1 + self.x_feedback_weight * x_feedback_factor

    def sample_prior(self, n_samples=1):
        c_samples = D.Bernoulli(probs=torch.tensor([self.c1_prob])).sample((n_samples,))
        orientation_samples = torch.where(c_samples == 0.0, self.c1_loc, self.c2_loc)
        diff_g_orientations = orientation_samples - self.g_pref_orientations.reshape(
            1, -1
        )
        g_probs = self.g_tuning(diff_g_orientations)
        g_samples = D.Bernoulli(probs=g_probs).sample()
        x_feedback_factor = g_samples @ self.x_feedback_tuning(
            self.g_pref_orientations.unsqueeze(-1)
            - self.x_pref_orientations.unsqueeze(0)
        )
        x_scale = self.x_tuning(x_feedback_factor)
        x_samples = D.Exponential(rate=1 / x_scale).sample()
        i_loc = x_samples @ self.x_pfs.reshape(self.x_pfs.shape[0], -1)
        i_samples = D.Normal(loc=i_loc, scale=self.obs_noise).sample()
        samples_dict = {
            "c": c_samples,
            "g": g_samples,
            "g_prob": g_probs,
            "x": x_samples,
            "i": i_samples,
            "x_scale": x_scale,
            "x_feedback_factor": x_feedback_factor,
        }
        return samples_dict
