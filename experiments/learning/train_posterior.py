import gensn.distributions as G
import torch
from gensn.parameters import TransformedParameter
from gensn.variational import VariationalBound

import wandb
from task_transfer.evaluation.evaluate_generative_model import (
    compute_logl,
    compute_var_marginal,
    logl_mc_marginal_eval,
    vpost_prior_eval_criterion,
)
from task_transfer.ml_lib.data_loading import (
    build_dataloaders,
    build_dataloaders_from_samples_paths,
)
from task_transfer.ml_lib.model_building import (
    build_conc_rate_mlp,
    build_conditional,
    build_flow_model,
    build_joint_model,
)
from task_transfer.ml_lib.trainer_building import (
    build_conditional_trainer,
    build_vpost_prior_trainer,
    zero_avoid,
)


def train_sbvp(
    data_loader_args, posterior_args, trainer_args, use_wandb=False, dj_conn=None
):
    if use_wandb:
        wandb_run = wandb.init(
            project="task_transfer_train_sbvp",
            entity="walkerlab",
            config={**data_loader_args, **posterior_args, **trainer_args},
        )

    # get the FP_Samples and MLPCond_Samples dataloaders
    samples_train_loader, samples_val_loader, samples_test_loader = (
        build_dataloaders_from_samples_paths(
            response_samples_path=data_loader_args["sampled_responses_path"],
            obs_samples_path=data_loader_args["sampled_obs_path"],
            train_prop=data_loader_args["train_prop"],
            val_prop=data_loader_args["val_prop"],
            batch_size=trainer_args["batch_size"],
            seed=data_loader_args["data_seed"],
        )
    )

    response_sample, image_sample = next(iter(samples_train_loader))

    torch.manual_seed(posterior_args["seed"])
    if posterior_args["dist"] == "gamma":
        amortization_fn = build_conc_rate_mlp(
            in_features=image_sample.shape[1],
            out_features_core=response_sample.shape[1],  # TODO: parameterize this?
            out_features_conc=response_sample.shape[1],  # TODO: parameterize this?
            out_features_rate=response_sample.shape[1],  # TODO: parameterize this?
            n_layers=posterior_args["n_layers"],
            nonlin=posterior_args["nonlin"],
            dropout_rate=posterior_args["dropout_rate"],
            init_std=posterior_args["init_std"],
            nonneg_transform=posterior_args["nonneg_transform"],
            clamp_pre_conc=posterior_args["kwargs"]["clamp_pre_conc"],
            pre_conc_max=posterior_args["kwargs"]["pre_conc_max"],
            clamp_pre_rate=posterior_args["kwargs"]["clamp_pre_rate"],
            pre_rate_min=posterior_args["kwargs"]["pre_rate_min"],
        )
        model = build_conditional(cond_dist="gamma", likelihood=amortization_fn)
    else:
        raise NotImplementedError("Unknown posterior distribution")

    # TODO: Set response_dim based on the dataloader args
    response_dim = 0  # set via experimenter's knowledge of the dataloader
    image_dim = 1  # set via experimenter's knowledge of the dataloader

    add_eps_to_data = True if posterior_args["dist"] == "gamma" else False

    trainer = build_conditional_trainer(
        model=model,
        data_dim=response_dim,
        cond_dim=image_dim,
        lr=trainer_args["lr"],
        weight_decay=trainer_args["weight_decay"],
        early_stopping_threshold=trainer_args["early_stopping_threshold"],
        early_stopping_patience=trainer_args["early_stopping_patience"],
        eval_criterion=None,
        eval_interval=None,
        eval_params=None,
        logging_type="wandb" if use_wandb else "stdout",
        device=trainer_args["device"],
        model_display_name="sbvp",
        add_eps_to_data=add_eps_to_data,
        dj_conn=dj_conn,
    )

    # if use_wandb:
    #     wandb.watch(model, log="all", log_freq=100)

    trainer_output = trainer.train(
        model=model,
        train_loader=samples_train_loader,
        val_loader=samples_val_loader,
        n_epochs=trainer_args["n_epochs"],
        ping_dj=False if dj_conn is None else True,
        # watch_grad_norm=True,  # TODO: debug code. cleanup
    )
    tracker_output = trainer_output["tracker_output"]
    eval_output = trainer_output["eval_output"]

    with torch.no_grad():
        model.eval()
        train_ll_mean_sample, train_ll_sem_sample = compute_logl(
            model=model,
            data_loader=samples_train_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )
        val_ll_mean_sample, val_ll_sem_sample = compute_logl(
            model=model,
            data_loader=samples_val_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )
        test_ll_mean_sample, test_ll_sem_sample = compute_logl(
            model=model,
            data_loader=samples_test_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )

    # also evaluate on real data
    # first load the real data
    # it's important to do this here since we do not want to train the model on real data
    # loading the real data here provides a safeguard against accidentally training on real data
    real_train_loader, real_val_loader, real_test_loader = build_dataloaders(
        data_loader_args["data_fname"],
        data_loader_args["train_prop"],
        data_loader_args["val_prop"],
        trainer_args["batch_size"],
    )
    # now evaluate the model on the real data
    with torch.no_grad():
        model.eval()
        train_ll_mean_real, train_ll_sem_real = compute_logl(
            model=model,
            data_loader=real_train_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )
        val_ll_mean_real, val_ll_sem_real = compute_logl(
            model=model,
            data_loader=real_val_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )
        test_ll_mean_real, test_ll_sem_real = compute_logl(
            model=model,
            data_loader=real_test_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            normalize="none",
            add_eps_to_data_dim=add_eps_to_data,
        )
    return (
        model,
        train_ll_mean_real,
        train_ll_sem_real,
        val_ll_mean_real,
        val_ll_sem_real,
        test_ll_mean_real,
        test_ll_sem_real,
        train_ll_mean_sample,
        train_ll_sem_sample,
        val_ll_mean_sample,
        val_ll_sem_sample,
        test_ll_mean_sample,
        test_ll_sem_sample,
        tracker_output,
        eval_output,
    )


def train_vpost_prior(
    data_loader_args, model_args, trainer_args, use_wandb=False, dj_conn=None
):
    # train the variational posterior and the prior
    # keep the likelihood fixed (and generally pre-trained)
    if model_args["seed"] < 0:
        # in this pipeline, negative seeds are used for training PRIOR models from scratch
        # positive seeds are used for training loaded (pre-trained) PRIOR models
        torch.manual_seed(-model_args["seed"])
    if use_wandb:
        wandb_run = wandb.init(
            project="task_transfer",
            entity="walkerlab",
            config={**data_loader_args, **model_args, **trainer_args},
        )

    # build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        data_fname=data_loader_args["data_fname"],
        train_prop=data_loader_args["train_prop"],
        val_prop=data_loader_args["val_prop"],
        batch_size=trainer_args["batch_size"],
    )
    response_dim = 0  # set via experimenter's knowledge of the dataloader
    image_dim = 1  # set via experimenter's knowledge of the dataloader
    response_sample, image_sample = next(iter(train_loader))
    n_response_dims = response_sample.shape[1]
    n_image_dims = image_sample.shape[1]  # expects flattened image

    # build approximate posterior
    if model_args["post_dist_type"] == "gamma":
        amortization_fn = build_conc_rate_mlp(
            in_features=n_image_dims,
            out_features_core=n_response_dims,
            out_features_conc=n_response_dims,
            out_features_rate=n_response_dims,
            n_layers=model_args["post_n_layers"],
            nonlin=model_args["post_nonlin"],
            dropout_rate=model_args["post_dropout_rate"],
            init_std=model_args["post_init_std"],
            nonneg_transform=model_args["post_nonneg_transform"],
            clamp_pre_conc=model_args["post_kwargs"]["clamp_pre_conc"],
            pre_conc_max=model_args["post_kwargs"]["pre_conc_max"],
            clamp_pre_rate=model_args["post_kwargs"]["clamp_pre_rate"],
            pre_rate_min=model_args["post_kwargs"]["pre_rate_min"],
        )
        posterior = build_conditional(cond_dist="gamma", likelihood=amortization_fn)
    elif model_args["post_dist_type"] == "gamma_pt_si":
        # load pre-trained (pt) system identification (si) model as posterior
        posterior = torch.load(
            model_args["si_model_path"], map_location=trainer_args["device"]
        )
    elif model_args["post_dist_type"] == "gamma_pt_vp":
        # load pre-trained (pt) variational posterior (vp) model as posterior
        posterior = torch.load(
            model_args["vp_model_path"], map_location=trainer_args["device"]
        )
    else:
        raise NotImplementedError("Unknown posterior distribution")

    # build generative model
    # build prior
    if model_args["seed"] == 666:
        lam = torch.nn.Parameter(torch.ones(n_response_dims))
        prior = G.IndependentExponential(rate=lam)
    elif model_args["seed"] == -666:
        lam = TransformedParameter(torch.randn(n_response_dims) * 1e-3, torch.exp)
        prior = G.IndependentExponential(rate=lam)
    elif model_args["seed"] < 0:
        # build prior model and train from scratch
        prior = build_flow_model(
            dims=n_response_dims,
            flow_base_distribution=model_args["prior_model_base_dist"],
            flow_depth=model_args["prior_model_depth"],
            flow_nonlinearity=model_args["prior_model_nonlin"],
            flow_initial_nonlinearity=model_args["prior_model_initial_nonlin"],
            flow_final_nonlinearity=model_args["prior_model_final_nonlin"],
            affine_type=model_args["prior_model_affine_type"],
        )
    else:
        # load pre-trained prior model
        prior = torch.load(
            model_args["prior_model_path"], map_location=trainer_args["device"]
        )

    # load conditional model
    conditional = torch.load(
        model_args["likelihood_model_path"], map_location=trainer_args["device"]
    )
    joint = build_joint_model(prior, conditional)

    variational_model = VariationalBound(
        joint=joint,
        posterior=posterior,
        bound_type=trainer_args["bound_type"],
        n_samples=trainer_args["n_bound_samples"],
    )

    eval_interval = 1
    eval_params = {
        "response_dim": response_dim,
        "image_dim": image_dim,
        "reduction": "mean",
        "uncertainty": "sem",
        "normalize": "none",
        "unit": "nats",
        "add_eps_to_data_dim": zero_avoid(model_args["post_dist_type"]),
    }
    # build trainer
    trainer = build_vpost_prior_trainer(
        model=variational_model,
        data_dim=image_dim,
        n_bound_samples=trainer_args["n_bound_samples"],
        lr=trainer_args["lr"],
        weight_decay=trainer_args["weight_decay"],
        eval_criterion=vpost_prior_eval_criterion,
        eval_interval=eval_interval,
        eval_params=eval_params,
        early_stopping_threshold=trainer_args["early_stopping_threshold"],
        early_stopping_patience=trainer_args["early_stopping_patience"],
        logging_type="wandb" if use_wandb else "stdout",
        device=trainer_args["device"],
        model_display_name="vpost_prior",
        dj_conn=dj_conn,
    )

    # train the model
    trainer_output = trainer.train(
        model=variational_model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=trainer_args["n_epochs"],
        watch_grad_norm=False,
        ping_dj=False if dj_conn is None else True,
    )

    tracker_output = trainer_output["tracker_output"]
    eval_output = trainer_output["eval_output"]

    # evaluate the model
    # compute the following:
    # 1. variational bound
    # 2. marginalized log likelihood
    # 3. posterior log likelihood on recorded latent (neuron) data
    # 4. prior log likelihood on recorded latent (neuron) data

    n_eval_samples = 1000  # TODO: parameterize this?
    with torch.no_grad():
        # 1. variational bound
        variational_model.eval()
        train_var_marginal_mean, train_var_marginal_sem = compute_var_marginal(
            var_model=variational_model,
            data_loader=train_loader,
            data_dim=image_dim,
            n_samples=n_eval_samples,
            bound_type=trainer_args["bound_type"],
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )
        val_var_marginal_mean, val_var_marginal_sem = compute_var_marginal(
            var_model=variational_model,
            data_loader=val_loader,
            data_dim=image_dim,
            n_samples=n_eval_samples,
            bound_type=trainer_args["bound_type"],
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )
        test_var_marginal_mean, test_var_marginal_sem = compute_var_marginal(
            var_model=variational_model,
            data_loader=test_loader,
            data_dim=image_dim,
            n_samples=n_eval_samples,
            bound_type=trainer_args["bound_type"],
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )

        # 2. marginalized log likelihood
        train_marginal_obs_ll_mean, train_marginal_obs_ll_sem = logl_mc_marginal_eval(
            variational_model.joint,
            train_loader,
            data_dim=image_dim,
            mc_sample_size=n_eval_samples,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )

        val_marginal_obs_ll_mean, val_marginal_obs_ll_sem = logl_mc_marginal_eval(
            variational_model.joint,
            val_loader,
            data_dim=image_dim,
            mc_sample_size=n_eval_samples,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )

        test_marginal_obs_ll_mean, test_marginal_obs_ll_sem = logl_mc_marginal_eval(
            variational_model.joint,
            test_loader,
            data_dim=image_dim,
            mc_sample_size=n_eval_samples,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )

        # 3. posterior log likelihood on recorded latent (neuron) data
        train_post_ll_mean, train_post_ll_sem = compute_logl(
            model=variational_model.posterior,
            data_loader=train_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
            add_eps_to_data_dim=zero_avoid(model_args["post_dist_type"]),
        )
        val_post_ll_mean, val_post_ll_sem = compute_logl(
            model=variational_model.posterior,
            data_loader=val_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
            add_eps_to_data_dim=zero_avoid(model_args["post_dist_type"]),
        )
        test_post_ll_mean, test_post_ll_sem = compute_logl(
            model=variational_model.posterior,
            data_loader=test_loader,
            data_dim=response_dim,
            cond_dim=image_dim,
            device=trainer.device,
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
            add_eps_to_data_dim=zero_avoid(model_args["post_dist_type"]),
        )

        # 4. prior log likelihood on recorded latent (neuron) data
        train_prior_ll_mean, train_prior_ll_sem = compute_logl(
            joint.prior,
            train_loader,
            data_dim=response_dim,
            cond_dim=None,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )
        val_prior_ll_mean, val_prior_ll_sem = compute_logl(
            joint.prior,
            val_loader,
            data_dim=response_dim,
            cond_dim=None,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )
        test_prior_ll_mean, test_prior_ll_sem = compute_logl(
            joint.prior,
            test_loader,
            data_dim=response_dim,
            cond_dim=None,
            device=trainer_args["device"],
            reduction="mean",
            uncertainty="sem",
            normalize="none",
            unit="nats",
        )

    return (
        variational_model,
        train_var_marginal_mean,
        train_var_marginal_sem,
        val_var_marginal_mean,
        val_var_marginal_sem,
        test_var_marginal_mean,
        test_var_marginal_sem,
        train_marginal_obs_ll_mean,
        train_marginal_obs_ll_sem,
        val_marginal_obs_ll_mean,
        val_marginal_obs_ll_sem,
        test_marginal_obs_ll_mean,
        test_marginal_obs_ll_sem,
        train_post_ll_mean,
        train_post_ll_sem,
        val_post_ll_mean,
        val_post_ll_sem,
        test_post_ll_mean,
        test_post_ll_sem,
        train_prior_ll_mean,
        train_prior_ll_sem,
        val_prior_ll_mean,
        val_prior_ll_sem,
        test_prior_ll_mean,
        test_prior_ll_sem,
        tracker_output,
        eval_output,
    )
