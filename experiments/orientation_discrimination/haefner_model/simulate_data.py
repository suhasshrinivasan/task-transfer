import os
import pickle

import configs as cfg
import torch

from task_transfer.sampling_models.plotting import plot_cohen_task, plot_haefner_model
from task_transfer.utils.model_utils import build_haefner_model


def simulate_data(
    p_c,
    c1_psi,
    c2_psi,
    kappa,
    g_phi,
    delta,
    lam,
    x_phi,
    obs_sigma,
    obs_h,
    obs_w,
    n_samples,
    seed=42,
    data_fname="data.pkl",
    plotting_params={
        "dpi": 300,
        "fontsize": 16,
        "linewidth": 4,
        "tick_length": 6,
        "tick_width": 2,
        "task_figfname": "task.pdf",
        "prior_figfname": "prior.pdf",
        "cdist_figfname": "cdist.pdf",
        "g_figfname": "g.pdf",
        "x_figfname": "x.pdf",
        "xcorr_figfname": "xcorr.pdf",
        "xcorr_hist_figfname": "xcorr_hist.pdf",
        "xdist_figfname": "xdist.pdf",
    },
    **kwargs,
):
    """
    Simulates data using the HaefnerModel and saves the results to a file. Additionally, it generates and saves plots.

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
        n_samples (int): Number of samples to generate.
        seed (int, optional): Seed for random number generation. Default is 42.
        data_fname (str, optional): Filename for saving the generated data. Default is "data.pkl".
        plotting_params (dict, optional): Parameters for plotting and saving figures. Contains keys:
            - "dpi" (int): Dots per inch for the figures.
            - "fontsize" (int): Font size for the figures.
            - "linewidth" (int): Line width for the figures.
            - "tick_length" (int): Tick length for the axes.
            - "tick_width" (int): Tick width for the axes.
            - "task_figfname" (str): Filename for saving the task plot.
            - "prior_figfname" (str): Filename for saving the prior plot.
            - "cdist_figfname" (str): Filename for saving the c distribution plot.
            - "g_figfname" (str): Filename for saving the g plot.
            - "x_figfname" (str): Filename for saving the x plot.
            - "xcorr_figfname" (str): Filename for saving the x correlation plot.
            - "xcorr_hist_figfname" (str): Filename for saving the x correlation histogram plot.
            - "xdist_figfname" (str): Filename for saving the x distribution plot.
        **kwargs: Catch-all for additional arguments.

    Returns:
        simulated data (dict): Dictionary containing the generated data.
    """

    # if os.path.exists(data_fname):
    #     raise FileExistsError(
    #         f"Operation canceled: '{data_fname}' already exists and will not be overwritten."
    #     )

    # for key, fname in plotting_params.items():
    #     if key.endswith("_figfname") and os.path.exists(fname):
    #         raise FileExistsError(
    #             f"Operation canceled: '{fname}' already exists and will not be overwritten."
    #         )

    # Set seed
    torch.manual_seed(seed)

    print("Generating Cohen task and prior plots...")
    plot_cohen_task(
        p_c=p_c,
        c1_psi=c1_psi,
        c2_psi=c2_psi,
        dpi=plotting_params["dpi"],
        fontsize=plotting_params["fontsize"],
        linewidth=plotting_params["linewidth"],
        tick_length=plotting_params["tick_length"],
        tick_width=plotting_params["tick_width"],
        task_figfname=plotting_params["task_figfname"],
        prior_figfname=plotting_params["prior_figfname"],
        cdist_figfname=plotting_params["cdist_figfname"],
    )
    print("Cohen task and prior plots generated.")

    print("Building Haefner model...")
    model = build_haefner_model(
        p_c=p_c,
        c1_psi=c1_psi,
        c2_psi=c2_psi,
        kappa=kappa,
        g_phi=g_phi,
        delta=delta,
        lam=lam,
        x_phi=x_phi,
        obs_sigma=obs_sigma,
        obs_h=obs_h,
        obs_w=obs_w,
    )
    print("Haefner model built.")

    print(f"Sampling {n_samples} samples from the prior distribution...")
    samples_dict = model.sample_prior(n_samples=n_samples)
    print(f"{n_samples} samples generated.")

    print("Generating Haefner model plots...")
    plot_haefner_model(
        model,
        samples_dict,
        g_figname=plotting_params["g_figfname"],
        x_figname=plotting_params["x_figfname"],
        xcorr_figname=plotting_params["xcorr_figfname"],
        xcorr_hist_figname=plotting_params["xcorr_hist_figfname"],
        xdist_figname=plotting_params["xdist_figfname"],
        plot_corr=plotting_params.get("plot_corr", True),
    )
    print("Haefner model plots generated.")

    print(f"Saving generated data to {data_fname}...")
    with open(data_fname, "wb") as f:
        pickle.dump(samples_dict, f)
    print(f"Data saved to {data_fname}.")

    return samples_dict


def main():
    # BELOW COMMENTED OUT SINCE DATA IS ALREADY GENERATED
    # print("Simulating data for the original Haefner 2AFC task 1...")
    # simulate_data(**cfg.orginal_haefner_2afc_task1)
    # print("Simulation for task 1 completed.")

    # print("Simulating data for the original Haefner 2AFC task 2...")
    # simulate_data(**cfg.orginal_haefner_2afc_task2)
    # print("Simulation for task 2 completed.")

    # GENERATING FLAT PRIOR DATA
    # print("Simulating data for the flat prior")
    # simulate_data(**cfg.flat_haefner)
    # print("Simulation for flat prior completed.")

    # GENERATING LARGE FLAT PRIOR DATA
    # simulate_data(**cfg.flat_haefner_100k)

    # GENERATE TOY DATA
    # print("Simulating data for the toy Haefner model...")
    # simulate_data(**cfg.flat_toy_haefner)

    # GENERATE 1D TOY DATA
    # print("Simulating data for the 1D toy Haefner model...")
    # simulate_data(**cfg.flat_toy_1neuron_haefner)

    # print("Simulating data for the 2D toy Haefner model...")
    # simulate_data(**cfg.flat_toy_2neuron_haefner)

    # print("Simulating data for the 3D toy Haefner model...")
    # simulate_data(**cfg.flat_toy_3neuron_haefner)

    # print("Simulating data for the 3D toy Haefner model...")
    # simulate_data(**cfg.flat_toy_10neuron_haefner)

    # simulate_data(**cfg.haefner_model_1neuron_task1)
    # simulate_data(**cfg.haefner_model_1neuron_task2)

    # simulate_data(**cfg.haefner_model_2neuron_task1)
    # simulate_data(**cfg.haefner_model_2neuron_task2)

    # simulate_data(**cfg.haefner_model_4neuron_task1)
    # simulate_data(**cfg.haefner_model_4neuron_task2)

    # simulate_data(**cfg.haefner_model_1neuron_highdelta_task1)
    # simulate_data(**cfg.haefner_model_1neuron_highdelta_task2)

    # simulate_data(**cfg.haefner_model_2neuron_highdelta_task1)
    # simulate_data(**cfg.haefner_model_2neuron_highdelta_task2)

    # simulate_data(**cfg.haefner_model_4neuron_highdelta_task1)
    # simulate_data(**cfg.haefner_model_4neuron_highdelta_task2)


if __name__ == "__main__":
    main()
