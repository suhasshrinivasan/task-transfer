import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ..utils.math_utils import cos2_von_mises


def plot_cohen_task(
    p_c,
    c1_psi,
    c2_psi,
    dpi=300,
    fontsize=16,
    linewidth=4,
    tick_length=6,
    tick_width=2,
    task_figfname="task_design.pdf",
    prior_figfname="prior.pdf",
    cdist_figfname="class_distribution.pdf",
):
    # plot prior distribution of classes
    fig, ax = plt.subplots(dpi=300, figsize=(3, 4))
    x = [0, 1]
    height = [p_c, 1 - p_c]
    ax.bar(x=x, height=height, color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(["$C=1$", "$C=2$"], fontsize=fontsize)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=fontsize)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        width=tick_width,
        length=tick_length,
    )
    ax.set_ylabel("$P(C)$", fontsize=fontsize)
    ax.set_xlabel("Class", fontsize=fontsize)
    ax.set_title("Class distribution", fontsize=fontsize)
    sns.despine(ax=ax, trim=True)
    ax.spines[["bottom", "left"]].set_linewidth(tick_width)
    fig.savefig(cdist_figfname, bbox_inches="tight", transparent=True)

    # plot task design
    all_orientations = torch.linspace(0, torch.pi, steps=1000)
    # TODO: make a task object instead of defining the orientation density here
    c1_orientation_density = cos2_von_mises(all_orientations, c1_psi, 1.0)
    c2_orientation_density = cos2_von_mises(all_orientations, c2_psi, 1.0)
    fig_task, ax_task = plt.subplots(dpi=dpi)
    ax_task.plot(
        all_orientations,
        c1_orientation_density,
        color="red",
        label="$C=1$",
        linewidth=linewidth,
    )
    ax_task.plot(
        all_orientations,
        c2_orientation_density,
        color="blue",
        label="$C=2$",
        linewidth=linewidth,
    )
    # TODO: set xticks based on task object
    xticks = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_task.set_xticks(xticks)
    ax_task.set_xticklabels(f"{int(xtick)}$^\circ$" for xtick in xticks * 180 / np.pi)
    ax_task.set_xlabel("Orientation $\\theta$ ($^\circ$)", fontsize=fontsize)
    ax_task.set_ylabel("Density $P(\\theta|C)$", fontsize=fontsize)
    ax_task.legend(loc="upper right", fontsize=fontsize)
    ax_task.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_task.set_title("Task design", fontsize=fontsize)
    ax_task.spines["bottom"].set_linewidth(tick_width)
    sns.despine(ax=ax_task, trim=True)
    ax_task.spines["left"].set_visible(False)
    ax_task.set_yticks([])
    fig_task.savefig(task_figfname, bbox_inches="tight", transparent=True)

    # plot prior
    fig_prior, ax_prior = plt.subplots(dpi=dpi)
    ax_prior.plot(
        all_orientations,
        (c1_orientation_density + c2_orientation_density) / 2,
        color="orange",
        linewidth=linewidth,
        label="Prior",
    )
    ax_prior.set_xticks(xticks)
    ax_prior.set_xticklabels(f"{int(xtick)}$^\circ$" for xtick in xticks * 180 / np.pi)
    ax_prior.set_xlabel("Orientation $\\theta$ ($^\circ$)", fontsize=fontsize)
    ax_prior.set_ylabel("Density $P(\\theta)$", fontsize=fontsize)
    ax_prior.legend(loc="upper right", fontsize=fontsize)
    ax_prior.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_prior.set_title("Prior", fontsize=fontsize)
    ax_prior.set_ylim(*ax_task.get_ylim())
    ax_prior.spines["bottom"].set_linewidth(tick_width)
    sns.despine(ax=ax_prior, trim=True)
    ax_prior.spines["left"].set_visible(False)
    ax_prior.set_yticks([])
    fig_prior.savefig(prior_figfname, bbox_inches="tight", transparent=True)


def plot_haefner_model(
    model,
    samples_dict,
    dpi=300,
    fontsize=16,
    linewidth=4,
    tick_length=6,
    tick_width=2,
    g_figname="g_activity.pdf",
    x_figname="x_activity.pdf",
    xcorr_figname="xcorr.pdf",
    xcorr_hist_figname="xcorr_hist.pdf",
    xdist_figname="xdist.pdf",
    plot_corr=True,
):

    all_orientations = torch.linspace(0, torch.pi, steps=1000)
    c1_orientation_density = cos2_von_mises(all_orientations, model.c1_psi, 1.0)
    c2_orientation_density = cos2_von_mises(all_orientations, model.c2_psi, 1.0)

    _, ax_task = plt.subplots(dpi=dpi)
    ax_task.plot(
        all_orientations,
        c1_orientation_density,
        color="red",
        label="$C=1$",
        linewidth=linewidth,
    )
    ax_task.plot(
        all_orientations,
        c2_orientation_density,
        color="blue",
        label="$C=2$",
        linewidth=linewidth,
    )
    xticks = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_task.set_xticks(xticks)
    ax_task.set_xticklabels(f"{int(xtick)}$^\circ$" for xtick in xticks * 180 / np.pi)
    ax_task.set_xlabel("Orientation $\\theta$ ($^\circ$)", fontsize=fontsize)
    ax_task.set_ylabel("Density $P(\\theta|C)$", fontsize=fontsize)
    ax_task.legend(loc="upper right", fontsize=fontsize)
    ax_task.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_task.set_title("Task design", fontsize=fontsize)

    ax_task.spines["bottom"].set_linewidth(tick_width)
    sns.despine(ax=ax_task, trim=True)
    ax_task.spines["left"].set_visible(False)
    # remove all yticks
    ax_task.set_yticks([])

    all_orientations = torch.linspace(0, torch.pi, steps=1000)
    c1_orientation_density = cos2_von_mises(all_orientations, model.c1_psi, 1.0)
    c2_orientation_density = cos2_von_mises(all_orientations, model.c2_psi, 1.0)

    # first set the random seed
    fig_prior, ax_prior = plt.subplots(dpi=dpi)
    xticks = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_prior.set_xticks(xticks)
    ax_prior.set_xticklabels(f"{int(xtick)}$^\circ$" for xtick in xticks * 180 / np.pi)

    ax_g = ax_prior.twinx()
    ax_g.bar(
        model.g_phi,
        samples_dict["g_samples"].mean(dim=0),
        width=np.pi / model.n_g,
        color="gray",
        edgecolor="black",
        # alpha=0.5,
        label="Avg firing rate $g_i$",
    )

    ax_prior.plot(
        all_orientations,
        (c1_orientation_density + c2_orientation_density) / 2,
        color="orange",
        linewidth=linewidth,
        label="Prior $\\theta$",
        linestyle="dashed",
    )
    ax_prior.set_xlabel(
        "Orientation preference $\\psi^g$ ($^\circ$)", fontsize=fontsize
    )
    ax_prior.set_ylabel("Density $p(\\theta)$", fontsize=fontsize)
    ax_prior.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_prior.set_title("Activity of G and prior", fontsize=fontsize)
    ax_prior.set_ylim(*ax_task.get_ylim())
    ax_prior.spines[["bottom", "left"]].set_linewidth(tick_width)
    ax_prior.set_zorder(ax_g.get_zorder() + 1)
    ax_prior.patch.set_visible(False)
    ax_prior.legend(loc="upper left", fontsize=fontsize)
    sns.despine(ax=ax_prior, trim=True)

    ax_g.set_ylim(ax_prior.get_ylim())
    ax_g.spines["right"].set_visible(True)
    ax_g.spines["right"].set_linewidth(tick_width)
    ax_g.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_g.set_ylabel("Probability $P(g_i = 1)$", fontsize=fontsize)

    ax_g.legend(loc="upper right", fontsize=fontsize)
    # ax_g.set_yticks([0, 0.1, 0.2, 0.3])
    sns.despine(ax=ax_g, trim=True)
    ax_g.spines["right"].set_visible(True)
    ax_g.spines["left"].set_visible(False)
    fig_prior.savefig(g_figname, bbox_inches="tight", transparent=True)

    fig_prior_x, ax_prior_x = plt.subplots(dpi=dpi)

    xticks = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax_prior_x.set_xticks(xticks)
    ax_prior_x.set_xticklabels(
        f"{int(xtick)}$^\circ$" for xtick in xticks * 180 / np.pi
    )
    ax_prior_x.set_xlabel(
        "Orientation preference $\\psi^x$ ($^\circ$)", fontsize=fontsize
    )

    ax_g = ax_prior_x.twinx()
    ax_g.bar(
        model.x_phi,
        samples_dict["tau"].mean(dim=0),
        width=np.pi / model.n_x,
        color="indianred",
        edgecolor="darkred",
        # alpha=0.5,
        label="Avg firing rate $x_i$",
        # zorder=0,
        # alpha=0,
    )
    ax_g.set_ylim([1, 1.4])

    ax_prior_x.plot(
        all_orientations,
        (c1_orientation_density + c2_orientation_density) / 2,
        color="orange",
        linewidth=linewidth,
        label="Prior $\\theta$",
        linestyle="dashed",
        # zorder=2,
        # alpha=0.7
    )
    ax_prior_x.set_ylabel("Density $p(\\theta)$", fontsize=fontsize, color="darkorange")
    ax_prior_x.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_prior_x.set_title("Activity of X and prior", fontsize=fontsize)
    ax_prior_x.set_ylim(*ax_task.get_ylim())
    ax_prior_x.set_yticklabels(ax_prior_x.get_yticklabels(), color="darkorange")

    ax_prior_x.spines[["bottom", "left"]].set_linewidth(tick_width)

    ax_prior_x.set_zorder(ax_g.get_zorder() + 1)
    ax_prior_x.patch.set_visible(False)
    # ax_g.axhline(1, color="black", linestyle="--")
    ax_prior_x.legend(loc="upper left", fontsize=fontsize)
    sns.despine(ax=ax_prior_x, trim=True)

    ax_g.spines["right"].set_visible(True)
    ax_g.spines["right"].set_linewidth(tick_width)
    ax_g.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    ax_g.set_ylabel("Avg firing rate", fontsize=fontsize, color="darkred")
    sns.despine(ax=ax_g, trim=True)
    ax_g.spines["right"].set_visible(True)
    ax_g.spines["right"].set_linewidth(tick_width)
    ax_g.spines["left"].set_visible(False)
    ax_g.set_yticklabels(ax_g.get_yticklabels(), color="darkred")
    ax_g.legend(loc="upper right", fontsize=fontsize)
    fig_prior_x.savefig(x_figname, bbox_inches="tight", transparent=True)

    if plot_corr:
        # plot correlation matrix
        xcorr = torch.corrcoef(samples_dict["x_samples"].T)
        # consider only the lower triangle without the diagonal
        mask = torch.triu(torch.ones_like(xcorr), diagonal=-1)
        antimask = torch.tril(torch.ones_like(xcorr), diagonal=-1)
        antimasked = antimask * xcorr
        vmin = torch.min(antimasked).item()
        vmax = torch.max(antimasked).item()
        cbar_ticks = np.linspace(vmin, vmax, 5)
        fig_xcorr, ax_xcorr = plt.subplots(dpi=300)
        sns.heatmap(
            xcorr.numpy(),
            mask=mask.numpy(),
            cmap="YlGn",
            vmin=vmin,
            vmax=vmax,
            ax=ax_xcorr,
            cbar_kws={"ticks": cbar_ticks},
        )

        ax_xcorr.set_xticks(np.arange(0, model.n_x, 11))
        ax_xcorr.set_yticks(np.arange(0, model.n_x, 11))

        xticklabels = ax_xcorr.set_xticklabels(
            [f"${int(np.rad2deg(x))}^\\circ$" for x in model.x_phi[::11]]
        )
        yticklabels = ax_xcorr.set_yticklabels(
            [f"${int(np.rad2deg(x))}^\\circ$" for x in model.x_phi[::11]]
        )

        cbar = ax_xcorr.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticklabels([f"{x * 100:.1f}%" for x in cbar_ticks])
        ax_xcorr.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            length=tick_length,
            width=tick_width,
        )

        ax_xcorr.set_ylabel("Preferred orientation $\\psi^x$", fontsize=fontsize)
        ax_xcorr.set_xlabel("Preferred orientation $\\psi^x$", fontsize=fontsize)
        ax_xcorr.set_title("Pearson correlation of $x$ (prior)", fontsize=fontsize)

        fig_xcorr.savefig(xcorr_figname, bbox_inches="tight", transparent=True)

        fig_xcorr_hist, ax_xcorr_hist = plt.subplots(dpi=dpi)
        sns.histplot(
            torch.masked_select(xcorr, antimask.bool()),
            ax=ax_xcorr_hist,
            stat="probability",
            element="step",
            # kde=True,
            color="green",
            # label="No task",
            alpha=0.5,
            label="",
        )
        ax_xcorr_hist.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
            length=tick_length,
            width=tick_width,
        )
        ax_xcorr_hist.set_xlabel(
            "Pearson correlation of $x$ (prior)", fontsize=fontsize
        )
        ax_xcorr_hist.set_ylabel("Probability", fontsize=fontsize)
        ax_xcorr_hist.set_xticks(cbar_ticks)
        ax_xcorr_hist.set_xticklabels([f"{x * 100:.1f}%" for x in cbar_ticks])

        sns.despine(ax=ax_xcorr_hist, trim=True)
        # ax_xcorr_hist.spines['left'].set_visible(False)
        ax_xcorr_hist.spines[["bottom", "left"]].set_linewidth(tick_width)

        fig_xcorr_hist.savefig(
            xcorr_hist_figname, bbox_inches="tight", transparent=True
        )

        ids = np.arange(0, model.n_x, step=model.n_x / 4).astype("int")

        fig_xdist, axs_dist = plt.subplots(2, 2, dpi=dpi, sharex=True, sharey=True)
        for ax, idx in zip(axs_dist.flatten(), ids):
            sns.histplot(
                samples_dict["x_samples"][:, idx],
                ax=ax,
                stat="density",
                element="step",
                # kde=True,
                color="darkorange",
                # label="No task",
                alpha=0.8,
                # linewidth=linewidth,
            )
            ax.text(
                0.7,
                0.9,
                f"$\\Psi^x=${int(np.rad2deg(model.x_phi[idx]))}$^\circ$",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize,
            )
            mean_samples = torch.mean(samples_dict["x_samples"][:, idx])
            ax.axvline(
                mean_samples, color="red", linestyle="dotted", linewidth=linewidth
            )
            ax.text(
                0.7,
                0.7,
                f"$\\lambda=${mean_samples:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize,
                color="red",
                label="Mean",
            )
            ax.set_xlim([0, 6])
            ax.set_ylim([0, 1])
            ax.set_xticks(np.arange(0, 7, step=3))
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=fontsize,
                length=tick_length,
                width=tick_width,
            )
            ax.set_ylabel("Density", fontsize=fontsize)
            ax.set_xlabel("Firing rate", fontsize=fontsize)
            sns.despine(ax=ax)
            ax.spines[["left", "bottom"]].set_linewidth(tick_width)
            # ax.axis("equal")
        fig_xdist.suptitle("Prior distribution of $x$ firing rates", fontsize=fontsize)
        fig_xdist.savefig(xdist_figname, bbox_inches="tight", transparent=True)
