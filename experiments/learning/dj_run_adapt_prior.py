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
# best_val_prior_results = fetch_best_model_results(
#     result_table=FlowPriorResult,
#     config_table=FlowPriorConfig,
#     data_loader_config_table=DataLoaderConfig,
#     trainer_config_table=FPTrainerConfig,
#     config_proj_col=prior_config_proj_col,
#     criterion=criterion,
#     k=k,
#     download_path=download_path,
# )

# get a prior model that does not model correlations
no_corr_restriction = "flow_base_dist = 'normal' and affine_type = 'factorized'"
flow_prior_config_table = FlowPriorConfig & no_corr_restriction
best_val_prior_results = fetch_best_model_results(
    result_table=FlowPriorResult,
    config_table=flow_prior_config_table,
    data_loader_config_table=DataLoaderConfig,
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
    data_loader_config_table=DataLoaderConfig,
    trainer_config_table=LLTrainerConfig,
    config_proj_col=likelihood_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)

AdaptPriorConfig.insert(
    [
        dict(
            seed=42,
            prior_fp_id=best_val_prior_results["fp_id"],
            prior_trainer_id=best_val_prior_results["trainer_id"],
            likelihood_id=best_val_likelihood_results["ll_id"],
            likelihood_trainer_id=best_val_likelihood_results["trainer_id"],
            orig_dl_id=best_val_prior_results["dl_id"],
        ),
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
    ],
    skip_duplicates=True,
)

dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_2_dataset.pkl",
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl",
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)

dataloader_configs_list = dict_product(dataloader_configs, insert_hash=True)

AltDataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)


trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[300],
    batch_size=[128],
    early_stopping_threshold=[1000],
    early_stopping_patience=[1000],
    mc_sample_size=[100],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

AdaptPriorTrainer.insert(trainer_configs_list, skip_duplicates=True)

AdaptPriorResult.USE_WANDB = True
AdaptPriorResult.FORCE_GPU = True

AdaptPriorResult.populate(
    "trainer_id = 'e267b2071bca2c3f9431f155e8e58f23' and seed = -100",
    reserve_jobs=True,
    suppress_errors=True,
)

# AdaptPriorResult.populate(order="original", limit=1)

# AdaptPriorResult.populate(
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )
