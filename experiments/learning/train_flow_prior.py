from task_transfer.evaluation.evaluate_generative_model import evaluate_flow_prior
from task_transfer.ml_lib.data_loading import build_dataloaders
from task_transfer.ml_lib.model_building import build_flow_model
from task_transfer.ml_lib.trainer_building import build_flow_trainer


def train_flow_prior(data_loader_args, prior_args, trainer_args):
    train_loader, val_loader, _ = build_dataloaders(
        data_fname=data_loader_args["data_fname"],
        train_prop=data_loader_args["train_prop"],
        val_prop=data_loader_args["val_prop"],
        batch_size=trainer_args["batch_size"],
    )

    response_sample, _ = next(iter(train_loader))
    n_prior_dims = response_sample.shape[1]
    flow_model = build_flow_model(
        dims=n_prior_dims,
        flow_base_distribution=prior_args["flow_base_dist"],
        flow_depth=prior_args["flow_depth"],
        flow_nonlinearity=prior_args["flow_nonlin"],
        flow_initial_nonlinearity=prior_args["flow_initial_nonlin"],
        flow_final_nonlinearity=prior_args["flow_final_nonlin"],
        affine_type=prior_args["affine_type"],
    )

    # TODO: Set response_dim based on the dataloader
    response_dim = 0  # set via knowledge of the dataloader
    trainer = build_flow_trainer(
        flow_model=flow_model,
        data_dim=response_dim,
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
        model=flow_model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=trainer_args["n_epochs"],
    )
    tracker_output = trainer_output["tracker_output"]
    eval_output = trainer_output["eval_output"]

    train_ll_mean, train_ll_sem = evaluate_flow_prior(
        flow=flow_model, data_loader=train_loader
    )
    val_ll_mean, val_ll_sem = evaluate_flow_prior(
        flow=flow_model, data_loader=val_loader
    )
    test_ll_mean, test_ll_sem = evaluate_flow_prior(
        flow=flow_model, data_loader=val_loader
    )
    return (
        flow_model,
        train_ll_mean,
        train_ll_sem,
        val_ll_mean,
        val_ll_sem,
        test_ll_mean,
        test_ll_sem,
        tracker_output,
        eval_output,
    )
