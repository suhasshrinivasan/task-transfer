from collections import OrderedDict

from experiments.dj.dataloader_tables import AltDataLoaderConfig, DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.prior_tables import AdaptPriorConfig, FlowPriorConfig
from experiments.dj.result_tables import (
    AdaptPriorResult,
    FlowPriorResult,
    LikelihoodResult,
)
from experiments.dj.trainer_tables import (
    AdaptPriorTrainer,
    FPTrainerConfig,
    LLTrainerConfig,
)
from task_transfer.utils.utils import dict_product

# first extract best prior and likelihood models
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1

prior_config_proj_col = "fp_id"
# get a prior model that is fit on flat prior dataset
dataset_restriction = "id = 'b8379e7d6998fc94a08a9a3742eec12d'"
data_loader_config_table = DataLoaderConfig & dataset_restriction
best_val_prior_results = fetch_best_model_results(
    result_table=FlowPriorResult,
    config_table=FlowPriorConfig,
    data_loader_config_table=data_loader_config_table,
    trainer_config_table=FPTrainerConfig,
    config_proj_col=prior_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)

likelihood_config_proj_col = "ll_id"
best_val_likelihood_results = fetch_best_model_results(
    result_table=LikelihoodResult,
    config_table=LikelihoodConfig,
    data_loader_config_table=data_loader_config_table,
    trainer_config_table=LLTrainerConfig,
    config_proj_col=likelihood_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)

AdaptPriorConfig.insert(
    [
        dict(
            seed=100,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
        dict(
            seed=-100,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
        dict(
            seed=42,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
        dict(
            seed=-42,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
        dict(
            seed=666,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
        dict(
            seed=-666,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
    ],
    skip_duplicates=True,
)


trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[300],
    batch_size=[128],
    early_stopping_threshold=[1000],
    early_stopping_patience=[1000],
    mc_sample_size=[1_000, 10_000],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

AdaptPriorTrainer.insert(trainer_configs_list, skip_duplicates=True)

AdaptPriorResult.USE_WANDB = True
AdaptPriorResult.FORCE_GPU = True

# three restrictions
# 1. train only on flat prior dataset
# 2. choose models pre-trained only on flat prior dataset
# 3. choose 1000 MC samples
# the last one is optional but helps save time

# restrictions = (
#     "and orig_dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and trainer_id = 'a7e83afb3d10e49d76cbfbe16c294932' "
# )

# AdaptPriorResult.populate(
#     restrictions,
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )


# four restrictions
# 1. train only on flat prior dataset
# 2. choose models pre-trained only on flat prior dataset
# 3. choose 10,000 MC samples
# 4. train only exp models, i.e., seed = +/- 666
# the last one is optional but helps save time

# restrictions = (
#     "(seed = 666 or seed = -666) "
#     "and orig_dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and trainer_id = 'b4400f97d2c4a40c100d69b05687bac2' "
# )

# AdaptPriorResult.populate(
#     restrictions,
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )


# four restrictions
# 1. train only on flat prior dataset
# 2. choose models pre-trained only on flat prior dataset
# 3. choose 10,000 MC samples
# 4. train only flow models
# the last one is optional but helps save time

# restrictions = (
#     "orig_dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
#     "and trainer_id = 'b4400f97d2c4a40c100d69b05687bac2' "
# )

# AdaptPriorResult.populate(
#     restrictions,
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )


# four restrictions
# 1. train only on flat prior 100k dataset
# 2. choose the following types of models:
#    - exp models (seed = +/- 666)
#    - randomly initialized flow model (seed < 0)
#    - pre-trained flow model on 10k flat prior dataset
# 3. choose 1,000 MC samples

restrictions = (
    "orig_dl_id = 'b8379e7d6998fc94a08a9a3742eec12d' "
    "and dl_id = 'd6b36dc9d4024882e4b7ccc597495a32' "
    "and trainer_id = 'a7e83afb3d10e49d76cbfbe16c294932'"
)
AdaptPriorResult.populate(
    restrictions,
    reserve_jobs=True,
    order="random",
    suppress_errors=True,
)
