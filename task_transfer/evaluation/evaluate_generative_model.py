from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def visualize_marginal_flow(
    flow,
    data_loader,
    epoch,
    device="cpu",
    density_support=(1e-3, 10),
    density_n_samples=1000,
    dims_to_plot=range(10),
    fig_dpi=300,
    linewidth=3,
    fontsize=10,
    plot_xlim=(0, 7),
    density_color="darkblue",
    data_color="darkorange",
    data_alpha=0.4,
    fig_save_dir=Path("/src/project/figures/learning/"),
    **catch_all,
):
    all_responses = []
    for batch in data_loader:
        responses, _ = batch
        all_responses.append(responses.to(device))
    all_responses = torch.cat(all_responses, dim=0)
    n_dims_to_plot = len(dims_to_plot)
    with torch.no_grad():
        flow.eval()
        n_dims_all = all_responses.shape[1]
        x = (
            torch.linspace(density_support[0], density_support[1], density_n_samples)
            .repeat(n_dims_all, 1)
            .T
        )
        flow_density = flow.factorized_log_prob(x).exp()
        fig, axs = plt.subplots(
            1,
            n_dims_to_plot,
            figsize=(6 * n_dims_to_plot, 4),
            sharey=True,
            dpi=fig_dpi,
        )
        for idx, ax in zip(dims_to_plot, axs):
            ax.plot(
                x[:, idx],
                flow_density[:, idx],
                linewidth=linewidth,
                color=density_color,
            )
            sns.histplot(
                all_responses[:, idx],
                ax=ax,
                stat="density",
                element="step",
                color=data_color,
                alpha=data_alpha,
            )
            ax.tick_params(axis="both", which="both", labelsize=fontsize)
            ax.set_ylabel("$p(x)$", fontsize=fontsize)
            ax.set_xlabel("x", fontsize=fontsize)
            ax.set_xlim(*plot_xlim)
        fig.savefig(
            fig_save_dir / f"{epoch}.pdf", bbox_inches="tight", transparent=True
        )
