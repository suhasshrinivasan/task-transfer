from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy.stats import kendalltau, pearsonr, spearmanr

from ..ml_lib.loss_criteria import (
    conditional_nll,
    joint_nll,
    marginal_nll,
    mc_marginal_nll,
)
from ..utils.utils import compute_uncertainty, convert_unit, normalize_tensor, reduce


def compute_logl(
    model,
    data_loader,
    data_dim,
    cond_dim=None,
    device="cpu",
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
    add_eps_to_data_dim=False,
):
    if cond_dim is None:
        logl_fn = lambda model, batch, data_dim, cond_dim: marginal_nll(
            model, batch, data_dim
        )  # here the model is the marginal model
    else:
        logl_fn = lambda model, batch, data_dim, cond_dim: conditional_nll(
            model, batch, data_dim, cond_dim, add_eps_to_data_dim
        )  # here the model is the conditional model
    log_probs = []
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            log_prob = -logl_fn(model, batch, data_dim, cond_dim)
            log_probs.append(log_prob)
    lp = reduce(log_probs, reduction)
    lp_uncertainty = compute_uncertainty(log_probs, uncertainty)
    data_sample = batch[data_dim]
    lp = normalize_tensor(lp, normalize, data_sample)
    lp = convert_unit(lp, unit)
    return lp, lp_uncertainty


def compute_var_marginal(
    var_model,
    data_loader,
    data_dim,
    n_samples,
    bound_type,
    device="cpu",
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
):
    log_probs = []
    with torch.no_grad():
        var_model.eval()
        var_model = var_model.to(device)
        for batch in data_loader:
            data = batch[data_dim].to(device)
            log_prob = -var_model.compute_bound(
                data, n_samples=n_samples, bound_type=bound_type
            )
            log_probs.append(log_prob)
    lp = reduce(log_probs, reduction)
    lp_uncertainty = compute_uncertainty(log_probs, uncertainty)
    lp = normalize_tensor(lp, normalize, data)
    lp = convert_unit(lp, unit)
    return lp, lp_uncertainty


def compute_logl_data_marginal(
    conditional_model,
    data_loader,
    data_dim,
    cond_dim,
    device="cpu",
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
):
    log_probs = []
    with torch.no_grad():
        conditional_model.eval()
        conditional_model = conditional_model.to(device)
        for batch in data_loader:
            data = batch[data_dim].to(device)
            cond = batch[cond_dim].to(device)
            conditional_ll = conditional_model(data, cond=cond.unsqueeze(1))
            marginal_ll = torch.logsumexp(conditional_ll, dim=0) - torch.log(
                torch.tensor(conditional_ll.shape[0])
            )
            log_probs.append(marginal_ll)
    lp = reduce(log_probs, reduction)
    lp_uncertainty = compute_uncertainty(log_probs, uncertainty)
    lp = normalize_tensor(lp, normalize, data)
    lp = convert_unit(lp, unit)
    return lp, lp_uncertainty


def compute_logl_marginal(
    conditional,
    prior,
    mc_sample_size,
    data_loader,
    data_dim,
    device,
    reduction,
    uncertainty,
    normalize,
    unit,
    prior_sampling_fn=None,
):
    log_probs = []
    with torch.no_grad():
        conditional = conditional.to(device)
        prior = prior.to(device)
        conditional.eval()
        prior.eval()
        for batch in data_loader:
            data = batch[data_dim]
            data = data.to(device)
            prior_sample = (
                prior.sample((mc_sample_size,))
                if prior_sampling_fn is None
                else prior_sampling_fn(prior, mc_sample_size)
            )
            conditional_ll = conditional(data, cond=prior_sample.unsqueeze(1))
            marginal_ll = torch.logsumexp(conditional_ll, dim=0) - torch.log(
                torch.tensor(conditional_ll.shape[0], device=device)
            )
            log_probs.append(marginal_ll)
        lp = reduce(log_probs, reduction)
        lp_uncertainty = compute_uncertainty(log_probs, uncertainty)
        lp = normalize_tensor(lp, normalize, data)
        lp = convert_unit(lp, unit)
    return lp, lp_uncertainty


def adapt_prior_eval_criterion(model, data_loader, epoch, device, eval_params, logger):
    prior_ll_mean, prior_ll_sem = compute_logl(
        model=model.prior,
        data_loader=data_loader,
        data_dim=eval_params["response_dim"],
        cond_dim=None,
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
    )
    joint_ll_mean, joint_ll_sem = compute_joint_logl(
        model=model,
        data_loader=data_loader,
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
    )
    cond_ll_mean, cond_ll_sem = compute_logl(
        model=model.conditional,
        data_loader=data_loader,
        data_dim=eval_params["image_dim"],
        cond_dim=eval_params["response_dim"],
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
    )
    metrics = {
        "eval/prior_ll_mean": prior_ll_mean,
        "eval/prior_ll_sem": prior_ll_sem,
        "eval/joint_ll_mean": joint_ll_mean,
        "eval/joint_ll_sem": joint_ll_sem,
        "eval/cond_ll_mean": cond_ll_mean,
        "eval/cond_ll_sem": cond_ll_sem,
    }
    track_message = f"Epoch {epoch} evaluation"
    logger.log(metrics, track_message)
    return metrics


def prior_eval_criterion(prior_model, data_loader, epoch, device, eval_params, logger):
    prior_ll_mean, prior_ll_sem = compute_logl(
        model=prior_model,
        data_loader=data_loader,
        data_dim=eval_params["response_dim"],
        cond_dim=None,
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
    )
    metrics = {
        "eval/prior_ll_mean": prior_ll_mean,
        "eval/prior_ll_sem": prior_ll_sem,
    }
    track_message = f"Epoch {epoch} evaluation"
    logger.log(metrics, track_message)
    return metrics


def conditional_eval_criterion(
    conditional_model, data_loader, epoch, device, eval_params, logger
):
    cond_ll_mean, cond_ll_sem = compute_logl(
        model=conditional_model,
        data_loader=data_loader,
        data_dim=eval_params["image_dim"],
        cond_dim=eval_params["response_dim"],
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
    )
    metrics = {"eval/cond_ll_mean": cond_ll_mean, "eval/cond_ll_sem": cond_ll_sem}
    track_message = f"Epoch {epoch} evaluation"
    logger.log(metrics, track_message)
    return metrics


def posterior_eval_criterion(
    posterior_model, data_loader, epoch, device, eval_params, logger
):
    posterior_ll_mean, posterior_ll_sem = compute_logl(
        model=posterior_model,
        data_loader=data_loader,
        data_dim=eval_params["response_dim"],
        cond_dim=eval_params["image_dim"],
        device=device,
        reduction=eval_params["reduction"],
        uncertainty=eval_params["uncertainty"],
        normalize=eval_params["normalize"],
        unit=eval_params["unit"],
        add_eps_to_data_dim=eval_params["add_eps_to_data_dim"],
    )
    metrics = {
        "eval/posterior_ll_mean": posterior_ll_mean,
        "eval/posterior_ll_sem": posterior_ll_sem,
    }
    track_message = f"Epoch {epoch} evaluation"
    logger.log(metrics, track_message)
    return metrics


def vpost_prior_eval_criterion(
    var_model, data_loader, epoch, device, eval_params, logger
):
    prior_metrics = prior_eval_criterion(
        var_model.joint.prior, data_loader, epoch, device, eval_params, logger
    )
    post_metrics = posterior_eval_criterion(
        var_model.posterior, data_loader, epoch, device, eval_params, logger
    )
    return {**prior_metrics, **post_metrics}


def compute_joint_logl(
    model,
    data_loader,
    device="cpu",
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
):
    log_probs = []
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            log_prob = -joint_nll(model, batch)
            log_probs.append(log_prob)
        if reduction == "mean":
            lp = torch.cat(log_probs).mean().item()
        elif reduction == "sum":
            lp = torch.cat(log_probs).sum().item()
        elif reduction == "none":
            lp = torch.cat(log_probs)
        else:
            raise ValueError("Unknown reduction")
        if uncertainty == "sem":
            lp_uncertainty = torch.cat(log_probs).std() / (len(log_probs) ** 0.5)
            lp_uncertainty = lp_uncertainty.item()
        elif uncertainty == "std":
            lp_uncertainty = torch.cat(log_probs).std()
            lp_uncertainty = lp_uncertainty.item()
        elif uncertainty == "none":
            lp_uncertainty = None
        else:
            raise ValueError("Unknown uncertainty measure")
        if normalize != "none":
            raise ValueError("Normalization not supported for joint log-likelihood")
        if unit == "nats":
            pass
        elif unit == "bits":
            lp /= np.log(2)
        else:
            raise ValueError("Unknown unit")
    return lp, lp_uncertainty


# TODO: use this
def logl_mc_marginal_eval(
    joint_model,
    data_loader,
    data_dim,
    mc_sample_size,
    device="cpu",
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
):
    log_probs = []
    mc_sample_size = (mc_sample_size,)
    with torch.no_grad():
        joint_model.eval()
        joint_model = joint_model.to(device)
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            log_prob = -mc_marginal_nll(joint_model, batch, data_dim, mc_sample_size)
            log_probs.append(log_prob)
        if reduction == "mean":
            lp = torch.cat(log_probs).mean().item()
        elif reduction == "sum":
            lp = torch.cat(log_probs).sum().item()
        elif reduction == "none":
            lp = torch.cat(log_probs)
        else:
            raise ValueError("Unknown reduction")
        if uncertainty == "sem":
            lp_uncertainty = torch.cat(log_probs).std() / (len(log_probs) ** 0.5)
            lp_uncertainty = lp_uncertainty.item()
        elif uncertainty == "std":
            lp_uncertainty = torch.cat(log_probs).std()
            lp_uncertainty = lp_uncertainty.item()
        elif uncertainty == "none":
            lp_uncertainty = None
        else:
            raise ValueError("Unknown uncertainty measure")
        if normalize == "per_dim":
            batch = next(iter(data_loader))
            data = batch[data_dim]
            lp /= data.shape[1:].numel()
        elif normalize == "none":
            pass
        elif normalize == "per_dim":
            n_dims = data.shape[1:].numel()
            lp /= n_dims
        if unit == "nats":
            pass
        elif unit == "bits":
            lp /= np.log(2)
        else:
            raise ValueError("Unknown unit")
    return lp, lp_uncertainty


# TODO: depricate in favor of logl_mc_marginal_eval


def logl_mc_marginal(
    joint_model,
    data_loader,
    data_dim,
    mc_sample_size,
    device="cpu",
):
    log_probs = []
    with torch.no_grad():
        joint_model.eval()
        joint_model = joint_model.to(device)
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            log_prob = -mc_marginal_nll(joint_model, batch, data_dim, mc_sample_size)
            log_probs.append(log_prob)
        mean_log_prob = torch.cat(log_probs).mean()
        sem_log_prob = torch.cat(log_probs).std() / (len(log_probs) ** 0.5)
    return mean_log_prob.item(), sem_log_prob.item()


# TODO: depricate in favor of compute_logl
def logl_conditional(
    model,
    data_loader,
    data_dim,
    cond_dim,
    device="cpu",
):
    log_probs = []
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        for batch in data_loader:
            batch = [b.to(device) for b in batch]
            log_prob = -conditional_nll(model, batch, data_dim, cond_dim)
            log_probs.append(log_prob)
        mean_log_prob = torch.cat(log_probs).mean()
        sem_log_prob = torch.cat(log_probs).std() / (len(log_probs) ** 0.5)
    return mean_log_prob.item(), sem_log_prob.item()


# TODO: depricate in favor of compute_logl
def logl_flow_prior(
    flow,
    data_loader,
    device="cpu",
):
    log_probs = []
    with torch.no_grad():
        flow.eval()
        flow = flow.to(device)
        for batch in data_loader:
            responses, _ = batch
            responses = responses.to(device)
            log_probs.append(flow(responses))
        mean_log_prob = torch.cat(log_probs).mean()
        sem_log_prob = torch.cat(log_probs).std() / (len(log_probs) ** 0.5)
    return mean_log_prob.item(), sem_log_prob.item()


def compute_haefner_logl_i_cond_x(
    haefner_model,
    data_loader,
    reduction="mean",
    uncertainty="sem",
    normalize="none",
    unit="nats",
):
    lps = []
    for responses, images in data_loader:
        lps.append(haefner_model.log_prob_i_cond_x(images, responses))
    lps = torch.cat(lps)
    if reduction == "mean":
        lp = lps.mean().item()
    elif reduction == "sum":
        lp = lps.sum().item()
    elif reduction == "none":
        lp = lps
    else:
        raise ValueError("Unknown reduction")
    if uncertainty == "sem":
        lp_uncertainty = lps.std() / (len(lps) ** 0.5)
        lp_uncertainty = lp_uncertainty.item()
    elif uncertainty == "std":
        lp_uncertainty = lps.std()
        lp_uncertainty = lp_uncertainty.item()
    elif uncertainty == "none":
        lp_uncertainty = None
    else:
        raise ValueError("Unknown uncertainty measure")
    if normalize == "none":
        pass
    elif normalize == "per_dim":
        n_dims = images.shape[1:].numel()
        lp /= n_dims
    if unit == "nats":
        pass
    elif unit == "bits":
        lp /= np.log(2)
    else:
        raise ValueError("Unknown unit")
    return lp, lp_uncertainty


def evaluate_correlation(model_corr, real_corr):
    sign_agreement_matrix = torch.where(
        torch.sign(model_corr) == torch.sign(real_corr),
        torch.ones_like(model_corr),
        torch.zeros_like(model_corr),
    )
    mse_matrix = (model_corr - real_corr) ** 2
    mae_matrix = (model_corr - real_corr).abs()

    tril_indices = torch.tril(torch.ones_like(model_corr), diagonal=-1).bool()
    model_corr_tril = model_corr[tril_indices]
    real_corr_tril = real_corr[tril_indices]

    pearson_corr, pearson_corr_p = pearsonr(model_corr_tril, real_corr_tril)

    spearman_corr, spearman_corr_p = spearmanr(model_corr_tril, real_corr_tril)

    kendall_tau, kendall_tau_p = kendalltau(model_corr_tril, real_corr_tril)

    sign_agreement_tril = sign_agreement_matrix[tril_indices]
    sign_agreement_mean = sign_agreement_tril.mean()
    sign_agreement_sem = sign_agreement_tril.std() / (len(sign_agreement_tril) ** 0.5)

    mse_tril = mse_matrix[tril_indices]

    mse_mean = mse_tril.mean()
    mse_sem = mse_tril.std() / (len(mse_matrix) ** 0.5)

    mae_tril = mae_matrix[tril_indices]
    mae_mean = mae_tril.mean()
    mae_sem = mae_tril.std() / (len(mae_matrix) ** 0.5)

    return (
        sign_agreement_mean,
        sign_agreement_sem,
        sign_agreement_matrix,
        mse_mean,
        mse_sem,
        mse_matrix,
        mae_mean,
        mae_sem,
        mae_matrix,
        pearson_corr,
        pearson_corr_p,
        spearman_corr,
        spearman_corr_p,
        kendall_tau,
        kendall_tau_p,
    )


def plot_marginal_hist(samples, all_responses, density_n_samples, plot_params):
    n_samples = samples.shape[0]
    n_hist_samples = min(n_samples, density_n_samples)
    # take n_hist_samples random samples from total samples
    n_hist_indices = torch.randperm(n_samples)[:n_hist_samples]
    hist_samples = samples[n_hist_indices]

    fig, axs = plt.subplots(7, 7, dpi=300, sharex=True, sharey=True)
    for idx, ax in zip(plot_params["dims_to_plot"], axs.flatten()):
        sns.histplot(
            all_responses[:, idx].detach().numpy(),
            ax=ax,
            stat="density",
            element="step",
            color=plot_params["data_color"],
            alpha=plot_params["data_alpha"],
            label="True",
            fill=True,
        )
        sns.histplot(
            hist_samples[:, idx].detach().numpy(),
            ax=ax,
            stat="density",
            element="step",
            color=plot_params["sample_color"],
            alpha=plot_params["sample_alpha"],
            label="Model",
            fill=False,
        )
        # ax.axis("off")
        ax.set_xlim(*plot_params["plot_xlim"])
        ax.set_ylim(*plot_params["plot_ylim"])
        sns.despine(ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(
            axis="both",
            which="major",
            length=0,
            width=0,
        )
    fig.supxlabel("$x$", fontsize=plot_params["fontsize"])
    fig.supylabel("$p(x)$", fontsize=plot_params["fontsize"])
    for ax in axs.flatten()[len(plot_params["dims_to_plot"]) :]:
        ax.axis("off")

    # make legend
    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=plot_params["fontsize"])
    fig.suptitle(
        "Marginal distributions of $x$ (prior)", fontsize=plot_params["fontsize"]
    )
    fig.tight_layout()

    fig.savefig(
        plot_params["fig_save_dir"] / "marginal_density.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()
    plt.close(fig)

    return fig


def plot_correlation_hists(sample_corr, real_corr, plot_params):

    antimask = torch.tril(torch.ones_like(real_corr), diagonal=-1)

    fig_xcorr_hist, ax_xcorr_hist = plt.subplots(dpi=plot_params["fig_dpi"])
    sns.histplot(
        torch.masked_select(sample_corr, antimask.bool()),
        ax=ax_xcorr_hist,
        stat="probability",
        element="step",
        # kde=True,
        color="darkblue",
        label="Model",
        fill=False,
        linewidth=plot_params["linewidth"],
    )

    sns.histplot(
        torch.masked_select(real_corr, antimask.bool()),
        ax=ax_xcorr_hist,
        stat="probability",
        element="step",
        fill=False,
        color="darkorange",
        label="Data",
        linewidth=plot_params["linewidth"],
    )
    ax_xcorr_hist.set_xlabel(
        "Pearson correlation (%)", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_hist.set_ylabel("Density", fontsize=plot_params["fontsize"])
    ax_xcorr_hist.tick_params(
        axis="both",
        which="major",
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )
    ax_xcorr_hist.legend(fontsize=plot_params["fontsize"])
    ax_xcorr_hist.set_title(
        "Correlation distributions of $x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_hist.set_xticks(ax_xcorr_hist.get_xticks())
    ax_xcorr_hist.set_xticklabels([f"{x*100:.1f}" for x in ax_xcorr_hist.get_xticks()])
    sns.despine(ax=ax_xcorr_hist, trim=True, offset=5)
    ax_xcorr_hist.spines[["left", "bottom"]].set_linewidth(plot_params["tick_width"])
    fig_xcorr_hist.savefig(
        plot_params["fig_save_dir"] / "correlation_hist.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()
    plt.close(fig_xcorr_hist)

    return fig_xcorr_hist


def correlation_map(corr, mask, vmin, vmax, cbar_ticks, title, plot_params):
    fig_xcorr_real, ax_xcorr_real = plt.subplots(dpi=plot_params["fig_dpi"])
    sns.heatmap(
        corr.numpy(),
        mask=mask.numpy(),
        cmap="PRGn",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
        ax=ax_xcorr_real,
        cbar_kws={"ticks": cbar_ticks},
    )

    # Adjust x and y ticks
    ax_xcorr_real.set_xticks(np.arange(0, 45, 11))
    ax_xcorr_real.set_yticks(np.arange(0, 45, 11))

    # Customize colorbar
    cbar = ax_xcorr_real.collections[0].colorbar
    cbar.set_ticklabels([f"{x * 100:.1f}%" for x in cbar_ticks])
    cbar.ax.tick_params(labelsize=plot_params["fontsize"])
    cbar.ax.tick_params(
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )

    # Customize axis ticks
    ax_xcorr_real.tick_params(
        axis="both",
        which="major",
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )

    # Set labels and title
    ax_xcorr_real.set_ylabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_real.set_xlabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_real.set_title(title, fontsize=plot_params["fontsize"])

    fig_xcorr_real.savefig(
        plot_params["fig_save_dir"] / "data_correlation.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    return fig_xcorr_real, ax_xcorr_real


def plot_correlation_maps(sample_corr, real_corr, plot_params):
    mask = torch.triu(torch.ones_like(real_corr), diagonal=0)
    antimask = torch.tril(torch.ones_like(real_corr), diagonal=-1)
    antimasked = antimask * real_corr

    # Compute the max absolute value for symmetric color limits
    abs_max = torch.max(torch.abs(antimasked)).item()
    vmin = -abs_max
    vmax = abs_max

    # Define color bar ticks
    cbar_ticks = np.linspace(vmin, vmax, 5)

    title = "Pearson correlation of $x$ (data)"
    fig_xcorr_data, ax_xcorr_data = correlation_map(
        real_corr, mask, vmin, vmax, cbar_ticks, title, plot_params
    )
    plt.show()
    plt.close(fig_xcorr_data)
    title = "Pearson correlation of $x$ (model)"
    fig_xcorr_sample, ax_xcorr_sample = correlation_map(
        sample_corr, mask, vmin, vmax, cbar_ticks, title, plot_params
    )
    plt.show()
    plt.close(fig_xcorr_sample)

    return fig_xcorr_data, fig_xcorr_sample


# def plot_correlation_diagnostics(
#     sign_agreement, mse, mae, pearson_corr, spearman_corr, kendall_corr, plot_params
# )
#     mask = torch.tril(torch.ones_like(sign_agreement), diagonal=0)


def plot_correlation_diagnostics(
    corr_sgn_matrix,
    corr_mse_matrix,
    corr_mae_matrix,
    plot_params,
):
    mask = torch.triu(torch.ones_like(corr_sgn_matrix), diagonal=0)

    fig_xcorr_sgn_agr, ax_xcorr_sgn_agr = plt.subplots(dpi=plot_params["fig_dpi"])

    # Define the colors for the colormap: dark red for 0, dark green for 1
    colors = ["lightgray", "green"]

    # Create the colormap
    cmap = ListedColormap(colors)
    sns.heatmap(
        corr_sgn_matrix.numpy(),
        mask=mask.numpy(),
        cmap=cmap,
        ax=ax_xcorr_sgn_agr,
        cbar=True,
    )

    # Adjust x and y ticks
    ax_xcorr_sgn_agr.set_xticks(np.arange(0, 45, 11))
    ax_xcorr_sgn_agr.set_yticks(np.arange(0, 45, 11))

    ax_xcorr_sgn_agr.tick_params(
        axis="both",
        which="major",
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )

    # Set labels and title
    ax_xcorr_sgn_agr.set_ylabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_sgn_agr.set_xlabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_sgn_agr.set_title(
        "Correlation sign agreement", fontsize=plot_params["fontsize"]
    )
    fig_xcorr_sgn_agr.savefig(
        plot_params["fig_save_dir"] / "xcorr_sgn_agr.pdf", bbox_inches="tight"
    )
    plt.show()
    plt.close(fig_xcorr_sgn_agr)

    mse_normed = (corr_mse_matrix - corr_mse_matrix.min()) / (
        corr_mse_matrix.max() - corr_mse_matrix.min()
    )
    # plot basic heatmaps for mse and mae
    fig_xcorr_mse, ax_xcorr_mse = plt.subplots(dpi=plot_params["fig_dpi"])
    sns.heatmap(
        mse_normed.numpy(),
        mask=mask.numpy(),
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax_xcorr_mse,
        cbar=True,
    )
    ax_xcorr_mse.set_xticks(np.arange(0, 45, 11))
    ax_xcorr_mse.set_yticks(np.arange(0, 45, 11))
    ax_xcorr_mse.tick_params(
        axis="both",
        which="major",
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )
    ax_xcorr_mse.set_ylabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_mse.set_xlabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_mse.set_title(
        "Mean squared error (min-max-normed)", fontsize=plot_params["fontsize"]
    )
    fig_xcorr_mse.savefig(
        plot_params["fig_save_dir"] / "mse.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()
    plt.close(fig_xcorr_mse)

    mae_normed = (corr_mae_matrix - corr_mae_matrix.min()) / (
        corr_mae_matrix.max() - corr_mae_matrix.min()
    )
    fig_xcorr_mae, ax_xcorr_mae = plt.subplots(dpi=plot_params["fig_dpi"])
    sns.heatmap(
        mae_normed.numpy(),
        mask=mask.numpy(),
        vmin=0,
        vmax=1,
        cmap="viridis",
        ax=ax_xcorr_mae,
        cbar=True,
    )
    ax_xcorr_mae.set_xticks(np.arange(0, 45, 11))
    ax_xcorr_mae.set_yticks(np.arange(0, 45, 11))
    ax_xcorr_mae.tick_params(
        axis="both",
        which="major",
        labelsize=plot_params["fontsize"],
        length=plot_params["tick_length"],
        width=plot_params["tick_width"],
    )
    ax_xcorr_mae.set_ylabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )
    ax_xcorr_mae.set_xlabel(
        "Preferred orientation $\\psi^x$", fontsize=plot_params["fontsize"]
    )

    ax_xcorr_mae.set_title(
        "Mean absolute error (min-max-normed)", fontsize=plot_params["fontsize"]
    )
    fig_xcorr_mae.savefig(
        plot_params["fig_save_dir"] / "mae.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()
    plt.close(fig_xcorr_mae)

    # return all figures
    return fig_xcorr_sgn_agr, fig_xcorr_mse, fig_xcorr_mae


def visualize_eval_correlation(samples, responses, plot_params):
    sample_corr = torch.corrcoef(samples.T)
    real_corr = torch.corrcoef(responses.T)

    (
        corr_sgn_agr_mean,
        corr_sgn_agr_sem,
        corr_sgn_matrix,
        corr_mse_mean,
        corr_mse_sem,
        corr_mse_matrix,
        corr_mae_mean,
        corr_mae_sem,
        corr_mae_matrix,
        corr_pearsonr,
        corr_personr_p,
        corr_spearmanr,
        corr_spearmanr_p,
        corr_kendalltau,
        corr_kendalltau_p,
    ) = evaluate_correlation(sample_corr, real_corr)

    fig_corr_hist = plot_correlation_hists(sample_corr, real_corr, plot_params)

    fig_corr_data, fig_corr_sample = plot_correlation_maps(
        sample_corr, real_corr, plot_params
    )

    fig_corr_sgn_agr, fig_corr_mse, fig_corr_mae = plot_correlation_diagnostics(
        corr_sgn_matrix,
        corr_mse_matrix,
        corr_mae_matrix,
        plot_params,
    )

    return (
        corr_sgn_agr_mean.numpy(),
        corr_sgn_agr_sem.numpy(),
        corr_sgn_matrix.numpy(),
        corr_mse_mean.numpy(),
        corr_mse_sem.numpy(),
        corr_mse_matrix.numpy(),
        corr_mae_mean.numpy(),
        corr_mae_sem.numpy(),
        corr_mae_matrix.numpy(),
        corr_pearsonr,
        corr_personr_p,
        corr_spearmanr,
        corr_spearmanr_p,
        corr_kendalltau,
        corr_kendalltau_p,
        fig_corr_hist,
        fig_corr_data,
        fig_corr_sample,
        fig_corr_sgn_agr,
        fig_corr_mse,
        fig_corr_mae,
    )


def evaluate_flow_prior(
    flow,
    data_loader,
    device="cpu",
    n_samples=100_000,
    density_n_samples=10_000,
    plot_params=dict(
        dims_to_plot=range(45),
        fig_dpi=300,
        linewidth=3,
        tick_length=6,
        tick_width=2,
        fontsize=10,
        plot_xlim=(0, 7),
        plot_ylim=(0, 1),
        density_color="darkblue",
        data_color="darkorange",
        data_alpha=1.0,
        sample_color="darkblue",
        sample_alpha=1.0,
        fig_save_dir=Path("/src/project/figures/learning/"),
    ),
    seed=42,
    **catch_all,
):
    torch.manual_seed(seed)
    with torch.no_grad():
        flow.eval()
        samples = flow.sample((n_samples,))
        flow = flow.to(device)
        all_responses = torch.cat(
            [responses.detach().to(device) for responses, _ in data_loader], dim=0
        )
        fig_marginal_hist = plot_marginal_hist(
            samples, all_responses, density_n_samples, plot_params
        )
        (
            corr_sgn_agr_mean,
            corr_sgn_agr_sem,
            corr_sgn_matrix,
            corr_mse_mean,
            corr_mse_sem,
            corr_mse_matrix,
            corr_mae_mean,
            corr_mae_sem,
            corr_mae_matrix,
            corr_pearsonr,
            corr_personr_p,
            corr_spearmanr,
            corr_spearmanr_p,
            corr_kendalltau,
            corr_kendalltau_p,
            fig_corr_hist,
            fig_corr_data,
            fig_corr_sample,
            fig_corr_sgn_agr,
            fig_corr_mse,
            fig_corr_mae,
        ) = visualize_eval_correlation(samples, all_responses, plot_params)
        # ensure all figures are closed to avoid memory leak
        plt.close(fig_corr_hist)
        plt.close(fig_corr_data)
        plt.close(fig_corr_sample)
        plt.close(fig_corr_sgn_agr)
        plt.close(fig_corr_mse)
        plt.close(fig_corr_mae)

        return (
            corr_sgn_agr_mean,
            corr_sgn_agr_sem,
            corr_sgn_matrix,
            corr_mse_mean,
            corr_mse_sem,
            corr_mse_matrix,
            corr_mae_mean,
            corr_mae_sem,
            corr_mae_matrix,
            corr_pearsonr,
            corr_personr_p,
            corr_spearmanr,
            corr_spearmanr_p,
            corr_kendalltau,
            corr_kendalltau_p,
            fig_corr_hist,
            fig_corr_data,
            fig_corr_sample,
            fig_corr_sgn_agr,
            fig_corr_mse,
            fig_corr_mae,
            fig_marginal_hist,
        )


def visualize_marginal_flow(
    flow,
    data_loader,
    epoch,
    device="cpu",
    density_support=(1e-3, 10),
    density_n_samples=1000,
    dims_to_plot=range(45),
    fig_dpi=300,
    linewidth=3,
    fontsize=10,
    plot_xlim=(0, 7),
    plot_ylim=(0, 1),
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
            7,
            7,
            sharey=True,
            dpi=fig_dpi,
        )
        for idx, ax in zip(dims_to_plot, axs.ravel()):
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
            ax.set_xlim(*plot_xlim)
            ax.set_ylim(*plot_ylim)
            ax.axis("off")
        # ax.tick_params(axis="both", which="both", labelsize=fontsize)
        # ax.set_ylabel("$p(x)$", fontsize=fontsize)
        # ax.set_xlabel("x", fontsize=fontsize)
        for ax in axs.ravel()[n_dims_to_plot:]:
            ax.axis("off")
        fig.savefig(
            fig_save_dir / f"{epoch}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        # close the figure to avoid memory leak
        plt.close(fig)


def visualize_conditional_features(
    conditional,
    data_loader,
    epoch,
    unit_perturbation=1,
    device="cpu",
    dims_to_plot=range(45),
    fig_dpi=300,
    fig_save_dir=Path("/src/project/figures/learning/"),
    **catch_all,
):
    response, image = next(iter(data_loader))
    image_len = image.shape[-1]
    h, w = int(image_len**0.5), int(image_len**0.5)
    fig, axs = plt.subplots(
        7,
        7,
        sharey=True,
        dpi=fig_dpi,
    )
    for dim, ax in zip(dims_to_plot, axs.ravel()):
        perturbation = torch.zeros_like(response[0])
        perturbation[dim] = unit_perturbation
        perturbation = perturbation.to(device)
        with torch.no_grad():
            conditional.eval()
            cond_dist = conditional.trainable_distribution.distribution(
                cond=perturbation
            )
            mean = cond_dist.mean
            ax.imshow(mean.reshape(h, w).cpu().numpy(), cmap="gray")
            ax.axis("off")
    for ax in axs.ravel()[len(dims_to_plot) :]:
        ax.axis("off")
    fig.savefig(
        fig_save_dir / f"{epoch}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    # close the figure to avoid memory leak
    plt.close(fig)


def evaluate_generative_model(
    model,
    data_loader,
    epoch,
    device="cpu",
    eval_params={
        "flow_params": {
            "density_support": (1e-3, 10),
            "density_n_samples": 1000,
            "dims_to_plot": range(10),
            "fig_dpi": 300,
            "linewidth": 3,
            "fontsize": 10,
            "plot_xlim": (0, 7),
            "density_color": "darkblue",
            "data_color": "darkorange",
            "data_alpha": 0.4,
            "fig_save_dir": Path("/src/project/figures/learning/marginal_density/"),
        },
        "conditional_params": {
            "unit_perturbation": 1,
            "dims_to_plot": range(10),
            "fig_dpi": 300,
            "fig_save_dir": Path("/src/project/figures/learning/conditional_features/"),
        },
    },
    **catch_all,
):
    visualize_marginal_flow(
        model.prior, data_loader, epoch, device=device, **eval_params["flow_params"]
    )
    visualize_conditional_features(
        model.conditional,
        data_loader,
        epoch,
        device=device,
        **eval_params["conditional_params"],
    )
    return


def visualize_marginal_flow(
    models,
    data_loader,
    device="cpu",
    density_support=(1e-3, 10),
    density_n_samples=1000,
    dims_to_plot=range(45),
    fig_dpi=300,
    linewidth=3,
    fontsize=10,
    plot_xlim=(0, 7),
    plot_ylim=(0, 1),
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
        n_dims_all = all_responses.shape[1]
        x = (
            torch.linspace(density_support[0], density_support[1], density_n_samples)
            .repeat(n_dims_all, 1)
            .T
        )
        fig, axs = plt.subplots(
            7,
            7,
            sharey=True,
            dpi=fig_dpi,
        )
        for idx, ax in zip(dims_to_plot, axs.ravel()):
            sns.histplot(
                all_responses[:, idx].detach().cpu(),
                ax=ax,
                stat="density",
                element="step",
                color=data_color,
                alpha=data_alpha,
                label="Data",
            )
        colors = sns.color_palette("tab10", n_colors=len(models))
        for model, color, label, linestyle in zip(
            models, colors, catch_all["labels"], catch_all["linestyles"]
        ):
            flow_density = model.factorized_log_prob(x.to(device)).exp().cpu().numpy()
            print(flow_density.shape)
            for idx, ax in zip(dims_to_plot, axs.ravel()):
                ax.plot(
                    x[:, idx].detach().cpu(),
                    flow_density[:, idx],
                    linewidth=linewidth,
                    color=color,
                    label=label,
                    linestyle=linestyle,
                )
                ax.set_xlim(*plot_xlim)
                ax.set_ylim(*plot_ylim)
                ax.axis("off")
            # ax.tick_params(axis="both", which="both", labelsize=fontsize)
            # ax.set_ylabel("$p(x)$", fontsize=fontsize)
            # ax.set_xlabel("x", fontsize=fontsize)
            for ax in axs.ravel()[n_dims_to_plot:]:
                ax.axis("off")
        handles, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=fontsize)
        fig.savefig(
            fig_save_dir / catch_all["fig_name"],
            bbox_inches="tight",
            transparent=True,
        )
        # close the figure to avoid memory leak
        plt.close(fig)
