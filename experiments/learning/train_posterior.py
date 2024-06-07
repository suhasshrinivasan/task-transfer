from task_transfer.evaluation.evaluate_generative_model import logl_conditional
from task_transfer.ml_lib.data_loading import build_dataloaders_from_samples_paths
from task_transfer.ml_lib.model_building import build_conc_rate_mlp, build_conditional
from task_transfer.ml_lib.trainer_building import build_conditional_trainer


def train_sbvp(data_loader_args, posterior_args, trainer_args):
    # get the FP_Samples and MLPCond_Samples dataloaders
    train_loader, val_loader, test_loader = build_dataloaders_from_samples_paths(
        response_samples_path=data_loader_args["response_samples_path"],
        obs_samples_path=data_loader_args["obs_samples_path"],
        train_prop=data_loader_args["train_prop"],
        val_prop=data_loader_args["val_prop"],
        batch_size=data_loader_args["batch_size"],
        seed=data_loader_args["seed"],
    )

    response_sample, image_sample = next(iter(train_loader))
    if posterior_args["dist"] == "gamma":
        amortization_fn = build_conc_rate_mlp(
            in_features=image_sample.shape[1],
            out_features_core=response_sample.shape[1],  # TODO: parameterize this?
            out_features_loc=response_sample.shape[1],  # TODO: parameterize this?
            out_features_scale=response_sample.shape[1],  # TODO: parameterize this?
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
        logging_type="stdout",
    )

    trainer_output = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=trainer_args["n_epochs"],
    )
    tracker_output = trainer_output["tracker_output"]
    eval_output = trainer_output["eval_output"]

    train_ll_mean, train_ll_sem = logl_conditional(
        model=model,
        data_loader=train_loader,
        data_dim=image_dim,
        cond_dim=response_dim,
        device=trainer.device,
    )
    val_ll_mean, val_ll_sem = logl_conditional(
        model=model,
        data_loader=val_loader,
        data_dim=image_dim,
        cond_dim=response_dim,
        device=trainer.device,
    )
    test_ll_mean, test_ll_sem = logl_conditional(
        model=model,
        data_loader=test_loader,
        data_dim=image_dim,
        cond_dim=response_dim,
        device=trainer.device,
    )
    return (
        model,
        train_ll_mean,
        train_ll_sem,
        val_ll_mean,
        val_ll_sem,
        test_ll_mean,
        test_ll_sem,
        tracker_output,
        eval_output,
    )
