# %%
import gensn.distributions as G
import torch
import torch.nn as nn

import experiments.orientation_discrimination.haefner_model.configs as data_cfg
from experiments.learning.adapt_prior import adapt_prior
from task_transfer.evaluation.evaluate_generative_model import (
    adapt_prior_eval_criterion,
    compute_joint_logl,
    compute_logl,
    compute_logl_data_marginal,
    compute_logl_marginal,
    evaluate_flow_prior,
    logl_mc_marginal_eval,
)
from task_transfer.ml_lib.data_loading import build_dataloaders
from task_transfer.utils.model_utils import build_haefner_model

# %%
cfg = data_cfg.flat_toy_2neuron_haefner
true_model = build_haefner_model(
    p_c=cfg["p_c"],
    c1_psi=cfg["c1_psi"],
    c2_psi=cfg["c2_psi"],
    kappa=cfg["kappa"],
    g_phi=cfg["g_phi"],
    delta=cfg["delta"],
    lam=cfg["lam"],
    x_phi=cfg["x_phi"],
    obs_sigma=cfg["obs_sigma"],
    obs_h=cfg["obs_h"],
    obs_w=cfg["obs_w"],
)


class TrueLikelihood(nn.Module):
    def __init__(self, in_features, out_features, weight_init, bias_init, sigma):
        super().__init__()
        self.fn = nn.Linear(in_features, out_features)
        self.fn.weight.data = weight_init
        self.fn.bias.data = bias_init
        self.sigma = sigma

    def forward(self, x):
        return self.fn(x), self.sigma


weight = true_model.x_pfs.view(true_model.x_pfs.shape[0], -1).T
bias = torch.zeros(weight.shape[0])
sigma = torch.nn.Parameter(torch.tensor(0.1))
true_likelihood = TrueLikelihood(weight.shape[1], weight.shape[0], weight, bias, sigma)
true_conditional = G.IndependentNormal(_parameters=true_likelihood)

likelihood_model_path = (
    "/src/project/data/synthetic/haefner_2afc/flat_toy_2neuron_haefner_likelihood.pt"
)
torch.save(true_conditional, likelihood_model_path)

data_loader_args = {
    "data_fname": "/src/project/data/synthetic/haefner_2afc/flat_toy_2neuron_haefner_dataset.pkl",
    "train_prop": 0.7,
    "val_prop": 0.2,
}

trainer_args = {
    "mc_sample_size": 10_000,
    "weight_decay": 1e-3,
    "lr": 1e-3,
    "early_stopping_patience": 100,
    "early_stopping_threshold": 100,
    "device": "cuda",
    "n_epochs": 2000,
    "batch_size": 128,
}

model_args = {
    "prior_model_base_dist": "normal",
    "prior_model_depth": 2,
    "prior_model_nonlin": "tanh",
    "prior_model_initial_nonlin": "inv_softplus",
    "prior_model_affine_type": "factorized",
    "prior_model_final_nonlin": "none",
    "seed": -42,
    "likelihood_model_path": likelihood_model_path,
}

# %%
true_prior = G.IndependentExponential(rate=torch.ones(2))

# %%
(
    joint_model,
    train_marginal_obs_ll_mean,
    train_marginal_obs_ll_sem,
    val_marginal_obs_ll_mean,
    val_marginal_obs_ll_sem,
    test_marginal_obs_ll_mean,
    test_marginal_obs_ll_sem,
    train_prior_ll_mean,
    train_prior_ll_sem,
    val_prior_ll_mean,
    val_prior_ll_sem,
    test_prior_ll_mean,
    test_prior_ll_sem,
    tracker_output,
    eval_output,
) = adapt_prior(
    data_loader_args=data_loader_args,
    model_args=model_args,
    trainer_args=trainer_args,
    use_wandb=False,
    dj_conn=None,
)

# %%
train_loader, val_loader, test_loader = build_dataloaders(
    data_fname=data_loader_args["data_fname"],
    train_prop=data_loader_args["train_prop"],
    val_prop=data_loader_args["val_prop"],
    batch_size=128,
)

# %%
data_dim = 1
cond_dim = 0
mc_sample_size = 50_000
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
reduction = "none"
uncertainty = "sem"
normalize = "none"
unit = "nats"
true_logl_marg_train, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=true_prior,
    mc_sample_size=mc_sample_size,
    data_loader=train_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
true_logl_marg_val, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=true_prior,
    mc_sample_size=mc_sample_size,
    data_loader=val_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
true_logl_marg_test, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=true_prior,
    mc_sample_size=mc_sample_size,
    data_loader=test_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)

pt_adpt_flow_logl_marg_train, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=joint_model.prior,
    mc_sample_size=mc_sample_size,
    data_loader=train_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
pt_adpt_flow_logl_marg_val, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=joint_model.prior,
    mc_sample_size=mc_sample_size,
    data_loader=val_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
pt_adpt_flow_logl_marg_test, _ = compute_logl_marginal(
    conditional=true_conditional,
    prior=joint_model.prior,
    mc_sample_size=mc_sample_size,
    data_loader=test_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = [
    true_logl_marg_train.detach().cpu(),
    pt_adpt_flow_logl_marg_train.detach().cpu(),
    true_logl_marg_val.detach().cpu(),
    pt_adpt_flow_logl_marg_val.detach().cpu(),
    true_logl_marg_test.detach().cpu(),
    pt_adpt_flow_logl_marg_test.detach().cpu(),
]
labels = [
    "True prior",
    "Flow adapted (PT)",
    "True prior",
    "Flow adapted (PT)",
    "True prior",
    "Flow adapted (PT)",
]
# Creating a boxplot
fig, ax = plt.subplots(dpi=300)
# set two colors as palette and cycle
palette = sns.color_palette("tab10", n_colors=9)
ax = sns.pointplot(
    data=data, ax=ax, errorbar="se", join=False, palette=palette, markers="."
)
# add mean values
# for i, value in enumerate(data):
#     # ax.text(i + 0.3, value.mean() + 0.5, f"{value.mean():.2f}", ha="center", va="bottom", fontsize=8)
#     # add a red dot
#     ax.plot(i, value.mean(), "rx")
ax.set_xticklabels(labels, rotation=90)
ax.set_title("Marginal image log-likelihood", fontsize=14)
ax.set_ylabel("Log-likelihood (nats)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=14)
for tick in range(len(labels)):
    if tick % 2 == 0 and tick != 0:
        ax.axvline(tick - 0.5, ls="--", color="grey")
# ax.set_yticks(range(-100, 0, 10))
sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig.savefig(
    "/src/project/figures/learning/marginal_logl2dlong2.pdf",
    bbox_inches="tight",
    transparent=True,
)

# # Display the plot
# plt.show()

# %%
data_dim = 0
device = "cuda"
reduction = "none"
uncertainty = "sem"
normalize = "none"
unit = "nats"

true_logl_prior_train, true_train_sem = compute_logl(
    model=true_prior,
    data_loader=train_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
true_logl_prior_val, true_val_sem = compute_logl(
    model=true_prior,
    data_loader=val_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
true_logl_prior_test, true_test_sem = compute_logl(
    model=true_prior,
    data_loader=test_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)

pt_adpt_flow_logl_prior_train, pt_adpt_flow_train_sem = compute_logl(
    model=joint_model.prior,
    data_loader=train_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
pt_adpt_flow_logl_prior_val, pt_adpt_flow_val_sem = compute_logl(
    model=joint_model.prior,
    data_loader=val_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)
pt_adpt_flow_logl_prior_test, pt_adpt_flow_test_sem = compute_logl(
    model=joint_model.prior,
    data_loader=test_loader,
    data_dim=data_dim,
    device=device,
    reduction=reduction,
    uncertainty=uncertainty,
    normalize=normalize,
    unit=unit,
)

# %%
data = [
    true_logl_prior_train.detach().cpu(),
    pt_adpt_flow_logl_prior_train.detach().cpu(),
    true_logl_prior_val.detach().cpu(),
    pt_adpt_flow_logl_prior_val.detach().cpu(),
    true_logl_prior_test.detach().cpu(),
    pt_adpt_flow_logl_prior_test.detach().cpu(),
]
labels = [
    "True prior",
    "Flow adapted (PT)",
    "True prior",
    "Flow adapted (PT)",
    "True prior",
    "Flow adapted (PT)",
]
# Creating a boxplot
fig, ax = plt.subplots(dpi=300)
# set two colors as palette and cycle
palette = sns.color_palette("tab10", n_colors=2)
ax = sns.pointplot(
    data=data, ax=ax, errorbar="se", join=False, palette=palette, markers="."
)
# add mean values
# for i, value in enumerate(data):
#     # ax.text(i + 0.3, value.mean() + 0.5, f"{value.mean():.2f}", ha="center", va="bottom", fontsize=8)
#     # add a red dot
#     ax.plot(i, value.mean(), "rx")
ax.set_xticklabels(labels, rotation=90)
ax.set_title("Prior log-likelihood", fontsize=14)
ax.set_ylabel("Log-likelihood (nats)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=14)
for tick in range(len(labels)):
    if tick % 2 == 0 and tick != 0:
        ax.axvline(tick - 0.5, ls="--", color="grey")

# ax.set_yticks(range(-300, 0, 30))
sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig.savefig(
    "/src/project/figures/learning/prior_logl2dlong2.pdf",
    bbox_inches="tight",
    transparent=True,
)

# # Display the plot
# plt.show()

# %%
from pathlib import Path


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
            1,
            2,
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


# %%


visualize_marginal_flow(
    models=[
        true_prior,
        joint_model.prior,
    ],
    data_loader=val_loader,
    device=device,
    density_support=(1e-3, 10),
    density_n_samples=1000,
    dims_to_plot=range(45),
    fig_dpi=300,
    linewidth=1,
    fontsize=10,
    plot_xlim=(0, 7),
    plot_ylim=(0, 1),
    data_color="lightpink",
    data_alpha=0.4,
    fig_save_dir=Path("/src/project/figures/learning/"),
    fig_name="prior2dlong2.pdf",
    labels=[
        "True prior",
        "Flow adapted (RND)",
    ],
    linestyles=["-", "--"],
)

# %%
tracker_output.keys()


# %%
def plot_training_results(trainer_output, plotting_params):
    """
    Plot the training results.

    Args:
        trainer_output (dict): Output dictionary from the trainer, has the following
            keys: "train_loss", "val_loss"
            values: List of training and validation losses.
    """
    # Plot training results
    train_losses = trainer_output["train_loss"]
    val_losses = trainer_output["val_loss"]
    fig, ax = plt.subplots(dpi=plotting_params["dpi"])
    ax.plot(
        train_losses,
        label="Train",
        marker="o",
        linestyle="--",
        color="indianred",
    )
    # plot with discrete markers

    ax.plot(val_losses, label="Validation", marker="o", linestyle="--", color="teal")
    ax.set_xlabel("Epochs", fontsize=plotting_params["fontsize"])
    ax.set_ylabel("NLL", fontsize=plotting_params["fontsize"])
    ax.set_title("Training and Validation NLL", fontsize=plotting_params["fontsize"])
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=plotting_params["fontsize"],
        length=plotting_params["tick_length"],
        width=plotting_params["tick_width"],
    )
    ax.legend(prop={"size": plotting_params["fontsize"]})

    ax.spines[["left", "bottom"]].set_linewidth(plotting_params["tick_width"])
    sns.despine(ax=ax, trim=True)
    fig.savefig(
        plotting_params["fig_save_dir"]
        / f"training_val_loss_{plotting_params['fname_suffix']}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


# %%
plotting_params = {
    "dpi": 300,
    "fontsize": 16,
    "linewidth": 4,
    "tick_length": 6,
    "tick_width": 2,
    "fig_save_dir": Path("/src/project/figures/learning/loss_curves/"),
    "fname_suffix": "2neuronlong2",
}
plot_training_results(tracker_output, plotting_params)

# %%
torch.save(
    joint_model,
    "/src/project/data/synthetic/haefner_2afc/flat_toy_2neuronlong_haefner_joint_model2.pt",
)
