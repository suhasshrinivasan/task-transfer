from pathlib import Path

from task_transfer.evaluation.evaluate_generative_model import evaluate_generative_model

generative_model_trainer_config = {
    "lr": 1e-3,
    "weight_decay": 0,
    "n_epochs": 200,
    "batch_size": 128,
    "early_stopping_threshold": 10,
    "early_stopping_patience": 10,
    "logging_type": "stdout",
    "train_prop": 0.6,
    "val_prop": 0.2,
    "eval_criterion": evaluate_generative_model,
    "eval_interval": 1,
    "eval_params": {
        "flow_params": {
            "density_support": (1e-3, 10),
            "density_n_samples": 1000,
            "dims_to_plot": range(45),
            "fig_dpi": 300,
            "linewidth": 2,
            "fontsize": 10,
            "plot_xlim": (0, 5),
            "plot_ylim": (0, 1),
            "density_color": "darkblue",
            "data_color": "darkorange",
            "data_alpha": 1,
            "fig_save_dir": Path("/src/project/figures/learning/marginal_density/"),
        },
        "conditional_params": {
            "unit_perturbation": 1,
            "dims_to_plot": range(45),
            "fig_dpi": 300,
            "fig_save_dir": Path("/src/project/figures/learning/conditional_features/"),
        },
        "loss_curve_params": {
            "dpi": 300,
            "fontsize": 16,
            "linewidth": 4,
            "tick_length": 6,
            "tick_width": 2,
            "fig_save_dir": Path("/src/project/figures/learning/loss_curves/"),
        },
    },
}

generative_model_trainer_config2 = {
    "lr": 5e-3,
    "weight_decay": 0,
    "n_epochs": 200,
    "batch_size": 128,
    "early_stopping_threshold": 100,
    "early_stopping_patience": 100,
    "logging_type": "stdout",
    "train_prop": 0.6,
    "val_prop": 0.2,
    "eval_criterion": None,
    "eval_interval": None,
    "eval_params": {
        "loss_curve_params": {
            "dpi": 300,
            "fontsize": 16,
            "linewidth": 4,
            "tick_length": 6,
            "tick_width": 2,
            "fig_save_dir": Path("/src/project/figures/learning/loss_curves/"),
        },
    },
}
