from pathlib import Path

import numpy as np
import torch

figdir = Path("/src/project/figures")
datadir = Path("/src/project/data/synthetic/haefner_2afc")

# Configuration parameters for original Haefner 2AFC Task 1 but with reduced n_g and n_x
orginal_haefner_2afc_task1 = {
    "p_c": 0.5,
    "c1_psi": np.pi / 4,
    "c2_psi": 3 * np.pi / 4,
    "kappa": 1.0,
    "g_phi": torch.linspace(0, np.pi, steps=9 + 1)[:-1],  # 9 instead of 256 original
    "delta": 1,
    "lam": 3.0,
    "x_phi": torch.linspace(0, np.pi, steps=45 + 1)[:-1],  # 45 instead of 1024 original
    "obs_sigma": 0.1,  # TODO: find out, not written in the paper
    "obs_h": 12,  # height of image
    "obs_w": 12,  # width of image
    "n_samples": 10_000,
    "seed": 42,
    "data_fname": datadir / "original_haefner_2afc_task_1_dataset.pkl",
    "plotting_params": {
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": figdir / "haefner_task_1.pdf",
        "prior_figfname": figdir / "haefner_prior_task_1.pdf",
        "cdist_figfname": figdir / "haefner_cdist_task_1.pdf",
        "g_figfname": figdir / "haefner_g_task_1.pdf",
        "x_figfname": figdir / "haefner_x_task_1.pdf",
        "xcorr_figfname": figdir / "haefner_xcorr_task_1.pdf",
        "xcorr_hist_figfname": figdir / "haefner_xcorr_hist_task_1.pdf",
        "xdist_figfname": figdir / "haefner_xdist_task_1.pdf",
    },
}

# Configuration parameters for original Haefner 2AFC Task 2
orginal_haefner_2afc_task2 = orginal_haefner_2afc_task1.copy()
orginal_haefner_2afc_task2["c1_psi"] = 0
orginal_haefner_2afc_task2["c2_psi"] = np.pi / 2
orginal_haefner_2afc_task2["data_fname"] = (
    datadir / "original_haefner_2afc_task_2_dataset.pkl"
)
orginal_haefner_2afc_task2["plotting_params"] = orginal_haefner_2afc_task1[
    "plotting_params"
].copy()
orginal_haefner_2afc_task2["plotting_params"].update(
    {
        "task_figfname": figdir / "haefner_task_2.pdf",
        "prior_figfname": figdir / "haefner_prior_task_2.pdf",
        "cdist_figfname": figdir / "haefner_cdist_task_2.pdf",
        "g_figfname": figdir / "haefner_g_task_2.pdf",
        "x_figfname": figdir / "haefner_x_task_2.pdf",
        "xcorr_figfname": figdir / "haefner_xcorr_task_2.pdf",
        "xcorr_hist_figfname": figdir / "haefner_xcorr_hist_task_2.pdf",
        "xdist_figfname": figdir / "haefner_xdist_task_2.pdf",
    }
)

flat_haefner = orginal_haefner_2afc_task1.copy()
flat_haefner["delta"] = 0
flat_haefner["data_fname"] = datadir / "flat_haefner_dataset.pkl"
flat_haefner["plotting_params"] = orginal_haefner_2afc_task1["plotting_params"].copy()
flat_haefner["plotting_params"].update(
    {
        "task_figfname": figdir / "flat_haefner_task.pdf",
        "prior_figfname": figdir / "flat_haefner_task_prior.pdf",
        "cdist_figfname": figdir / "flat_haefner_cdist.pdf",
        "g_figfname": figdir / "flat_haefner_g.pdf",
        "x_figfname": figdir / "flat_haefner_x.pdf",
        "xcorr_figfname": figdir / "flat_haefner_xcorr.pdf",
        "xcorr_hist_figfname": figdir / "flat_haefner_xcorr_hist.pdf",
        "xdist_figfname": figdir / "flat_haefner_xdist.pdf",
    }
)


flat_haefner_100k = orginal_haefner_2afc_task1.copy()
flat_haefner_100k["n_samples"] = 100_000
flat_haefner_100k["delta"] = 0
flat_haefner_100k["data_fname"] = datadir / "flat_haefner_100k_dataset.pkl"
flat_haefner_100k["plotting_params"] = orginal_haefner_2afc_task1[
    "plotting_params"
].copy()
flat_haefner_100k["plotting_params"].update(
    {
        "task_figfname": figdir / "flat_haefner_100k_task.pdf",
        "prior_figfname": figdir / "flat_haefner_100k_task_prior.pdf",
        "cdist_figfname": figdir / "flat_haefner_100k_cdist.pdf",
        "g_figfname": figdir / "flat_haefner_100k_g.pdf",
        "x_figfname": figdir / "flat_haefner_100k_x.pdf",
        "xcorr_figfname": figdir / "flat_haefner_100k_xcorr.pdf",
        "xcorr_hist_figfname": figdir / "flat_haefner_100k_xcorr_hist.pdf",
        "xdist_figfname": figdir / "flat_haefner_100k_xdist.pdf",
    }
)


flat_toy_3neuron_haefner = {
    "p_c": 0.5,
    "c1_psi": np.pi / 4,
    "c2_psi": 3 * np.pi / 4,
    "kappa": 1.0,
    "g_phi": torch.tensor([0.0, np.pi / 2]),
    "delta": 0,
    "lam": 3.0,
    "x_phi": torch.tensor(
        [0.0, np.pi / 4, 3 * np.pi / 4]
    ),  # 45 instead of 1024 original
    "obs_sigma": 0.1,  # TODO: find out, not written in the paper
    "obs_h": 12,  # height of image
    "obs_w": 12,  # width of image
    "n_samples": 10_000,
    "seed": 42,
    "data_fname": datadir / "flat_toy_3neuron_haefner_dataset.pkl",
    "plotting_params": {
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": figdir / "flat_toy_3neuron_haefner_task_1.pdf",
        "prior_figfname": figdir / "flat_toy_3neuron_haefner_prior_task_1.pdf",
        "cdist_figfname": figdir / "flat_toy_3neuron_haefner_cdist_task_1.pdf",
        "g_figfname": figdir / "flat_toy_3neuron_haefner_g_task_1.pdf",
        "x_figfname": figdir / "flat_toy_3neuron_haefner_x_task_1.pdf",
        "xcorr_figfname": figdir / "flat_toy_3neuron_haefner_xcorr_task_1.pdf",
        "xcorr_hist_figfname": figdir
        / "flat_toy_3neuron_haefner_xcorr_hist_task_1.pdf",
        "xdist_figfname": figdir / "flat_toy_3neuron_haefner_xdist_task_1.pdf",
    },
}

flat_toy_2neuron_haefner = {
    "p_c": 0.5,
    "c1_psi": np.pi / 4,
    "c2_psi": 3 * np.pi / 4,
    "kappa": 1.0,
    "g_phi": torch.tensor([0.0, np.pi / 2]),
    "delta": 0,
    "lam": 3.0,
    "x_phi": torch.tensor([0.0, np.pi / 2]),  # 45 instead of 1024 original
    "obs_sigma": 0.1,  # TODO: find out, not written in the paper
    "obs_h": 12,  # height of image
    "obs_w": 12,  # width of image
    "n_samples": 10_000,
    "seed": 42,
    "data_fname": datadir / "flat_toy_2neuron_haefner_dataset.pkl",
    "plotting_params": {
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": figdir / "flat_toy_2neuron_haefner_task_1.pdf",
        "prior_figfname": figdir / "flat_toy_2neuron_haefner_prior_task_1.pdf",
        "cdist_figfname": figdir / "flat_toy_2neuron_haefner_cdist_task_1.pdf",
        "g_figfname": figdir / "flat_toy_2neuron_haefner_g_task_1.pdf",
        "x_figfname": figdir / "flat_toy_2neuron_haefner_x_task_1.pdf",
        "xcorr_figfname": figdir / "flat_toy_2neuron_haefner_xcorr_task_1.pdf",
        "xcorr_hist_figfname": figdir
        / "flat_toy_2neuron_haefner_xcorr_hist_task_1.pdf",
        "xdist_figfname": figdir / "flat_toy_2neuron_haefner_xdist_task_1.pdf",
    },
}


flat_toy_1neuron_haefner = {
    "p_c": 0.5,
    "c1_psi": np.pi / 4,
    "c2_psi": 3 * np.pi / 4,
    "kappa": 1.0,
    "g_phi": torch.tensor([0.0, np.pi / 2]),
    "delta": 0,
    "lam": 3.0,
    "x_phi": torch.tensor([0.0]),  # 45 instead of 1024 original
    "obs_sigma": 0.1,  # TODO: find out, not written in the paper
    "obs_h": 12,  # height of image
    "obs_w": 12,  # width of image
    "n_samples": 10_000,
    "seed": 42,
    "data_fname": datadir / "flat_toy_1neuron_haefner_dataset.pkl",
    "plotting_params": {
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": figdir / "flat_toy_1neuron_haefner_task_1.pdf",
        "prior_figfname": figdir / "flat_toy_1neuron_haefner_prior_task_1.pdf",
        "cdist_figfname": figdir / "flat_toy_1neuron_haefner_cdist_task_1.pdf",
        "g_figfname": figdir / "flat_toy_1neuron_haefner_g_task_1.pdf",
        "x_figfname": figdir / "flat_toy_1neuron_haefner_x_task_1.pdf",
        "xcorr_figfname": figdir / "flat_toy_1neuron_haefner_xcorr_task_1.pdf",
        "xcorr_hist_figfname": figdir
        / "flat_toy_1neuron_haefner_xcorr_hist_task_1.pdf",
        "xdist_figfname": figdir / "flat_toy_1neuron_haefner_xdist_task_1.pdf",
        "plot_corr": False,
    },
}


flat_toy_10neuron_haefner = {
    "p_c": 0.5,
    "c1_psi": np.pi / 4,
    "c2_psi": 3 * np.pi / 4,
    "kappa": 1.0,
    "g_phi": torch.tensor([0.0, np.pi / 2]),
    "delta": 0,
    "lam": 3.0,
    "x_phi": torch.linspace(0, np.pi, steps=11)[:-1],
    "obs_sigma": 0.1,  # TODO: find out, not written in the paper
    "obs_h": 12,  # height of image
    "obs_w": 12,  # width of image
    "n_samples": 10_000,
    "seed": 42,
    "data_fname": datadir / "flat_toy_10neuron_haefner_dataset.pkl",
    "plotting_params": {
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": figdir / "flat_toy_10neuron_haefner_task_1.pdf",
        "prior_figfname": figdir / "flat_toy_10neuron_haefner_prior_task_1.pdf",
        "cdist_figfname": figdir / "flat_toy_10neuron_haefner_cdist_task_1.pdf",
        "g_figfname": figdir / "flat_toy_10neuron_haefner_g_task_1.pdf",
        "x_figfname": figdir / "flat_toy_10neuron_haefner_x_task_1.pdf",
        "xcorr_figfname": figdir / "flat_toy_10neuron_haefner_xcorr_task_1.pdf",
        "xcorr_hist_figfname": figdir
        / "flat_toy_10neuron_haefner_xcorr_hist_task_1.pdf",
        "xdist_figfname": figdir / "flat_toy_10neuron_haefner_xdist_task_1.pdf",
    },
}
