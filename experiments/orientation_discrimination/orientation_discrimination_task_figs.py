# DEPRICATED! USE
from pathlib import Path

import gensn
import gensn.distributions as G
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributions as D
from numpy.random import default_rng
from scipy.special import i0

from task_transfer.ml_lib.flow import get_flow_prior_model
from task_transfer.ml_lib.routines import lstsq_solution, mc_marginal_log_likelihood
from task_transfer.sampling_models.plotting import (
    plot_orientation_discrimination_model_prior,
    plot_orientation_discrimination_task,
)
from task_transfer.sampling_models.sampling_models import OrientationDiscriminationModel
from task_transfer.utils.insilico_stimuli import generate_gabors
from task_transfer.utils.math_utils import cos2_von_mises

seed = 1123
torch.manual_seed(seed)
rng = default_rng(seed)


n_g = 9
n_x = 45
c1_prob = 0.5
ta_c1_loc = np.pi / 4
tb_c1_loc = 0  # task B
c1_std = 1.0
ta_c2_loc = 3 * np.pi / 4
tb_c2_loc = np.pi / 2  # task B
c2_std = 1.0
g_pref_orientations = torch.linspace(0, np.pi, steps=n_g)[:-1]
# g_pref_orientations = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4])
g_tuning_specificity = 1.0
x_feedback_weight = 1
x_tuning_specificity = 3
x_pref_orientations = torch.linspace(g_pref_orientations[0], np.pi, steps=n_x)[:-1]
obs_noise = 0.1
n_samples = 10_000

fig_dir = Path("/src/project/figures/")

task_figname_a = fig_dir / "cohen_task_A.pdf"
prior_figname_a = fig_dir / "cohen_prior_task_A.pdf"
class_dist_figname_a = fig_dir / "cohen_class_dist_task_A.pdf"

g_figname_task_a = fig_dir / "cohen_g_task_A.pdf"
x_figname_task_a = fig_dir / "cohen_x_task_A.pdf"
x_corr_figname_task_a = fig_dir / "cohen_xcorr_task_A.pdf"
x_corr_hist_figname_task_a = fig_dir / "cohen_xcorr_hist_task_A.pdf"
xdist_figname_task_a = fig_dir / "cohen_xdist_task_A.pdf"


task_figname_b = fig_dir / "cohen_task_B.pdf"
prior_figname_b = fig_dir / "cohen_prior_task_B.pdf"
class_dist_figname_b = fig_dir / "cohen_class_dist_task_B.pdf"

g_figname_task_b = fig_dir / "cohen_g_task_B.pdf"
x_figname_task_b = fig_dir / "cohen_x_task_B.pdf"
x_corr_figname_task_b = fig_dir / "cohen_xcorr_task_B.pdf"
x_corr_hist_figname_task_b = fig_dir / "cohen_xcorr_hist_task_B.pdf"
xdist_figname_task_b = fig_dir / "cohen_xdist_task_B.pdf"


model_a = OrientationDiscriminationModel(
    c1_prob=c1_prob,
    c1_loc=ta_c1_loc,
    c1_std=c1_std,
    c2_loc=ta_c2_loc,
    c2_std=c2_std,
    g_pref_orientations=g_pref_orientations,
    g_tuning_specificity=g_tuning_specificity,
    x_feedback_weight=x_feedback_weight,
    x_tuning_specificity=x_tuning_specificity,
    x_pref_orientations=x_pref_orientations,
    obs_noise=obs_noise,
)
samples_dict_a = model_a.sample_prior(n_samples=n_samples)
model_b = OrientationDiscriminationModel(
    c1_prob=c1_prob,
    c1_loc=tb_c1_loc,
    c1_std=c1_std,
    c2_loc=tb_c2_loc,
    c2_std=c2_std,
    g_pref_orientations=g_pref_orientations,
    g_tuning_specificity=g_tuning_specificity,
    x_feedback_weight=x_feedback_weight,
    x_tuning_specificity=x_tuning_specificity,
    x_pref_orientations=x_pref_orientations,
    obs_noise=obs_noise,
)
samples_dict_b = model_b.sample_prior(n_samples=n_samples)

plot_orientation_discrimination_task(
    model_a,
    task_figname=task_figname_a,
    prior_figname=prior_figname_a,
    class_dist_figname=class_dist_figname_a,
)


plot_orientation_discrimination_model_prior(
    model_a,
    samples_dict_a,
    g_figname=g_figname_task_a,
    x_figname=x_figname_task_a,
    xcorr_figname=x_corr_figname_task_a,
    xcorr_hist_figname=x_corr_hist_figname_task_a,
    xdist_figname=xdist_figname_task_a,
)

plot_orientation_discrimination_task(
    model_b,
    task_figname=task_figname_b,
    prior_figname=prior_figname_b,
    class_dist_figname=class_dist_figname_b,
)

plot_orientation_discrimination_model_prior(
    model_b,
    samples_dict_b,
    g_figname=g_figname_task_b,
    x_figname=x_figname_task_b,
    xcorr_figname=x_corr_figname_task_b,
    xcorr_hist_figname=x_corr_hist_figname_task_b,
    xdist_figname=xdist_figname_task_b,
)
