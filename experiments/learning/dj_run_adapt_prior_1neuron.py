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
dataset_restriction = "id = 'f1ae78885d2ace1ba976199d4cf1a4d6'"
data_loader_config_table = DataLoaderConfig & dataset_restriction
best_val_prior_results = fetch_best_model_results(
    result_table_name=FlowPriorResult,
    config_table_name=FlowPriorConfig,
    data_loader_config_table_name=data_loader_config_table,
    trainer_config_table_name=FPTrainerConfig,
    config_proj_col=prior_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)

likelihood_config_proj_col = "ll_id"
best_val_likelihood_results = fetch_best_model_results(
    result_table_name=LikelihoodResult,
    config_table_name=LikelihoodConfig,
    data_loader_config_table_name=data_loader_config_table,
    trainer_config_table_name=LLTrainerConfig,
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

# training from scratch
# AdaptPriorResult.populate(
#     "trainer_id = 'e267b2071bca2c3f9431f155e8e58f23' and seed = -100",
#     reserve_jobs=True,
#     suppress_errors=True,
# )

# AdaptPriorResult.populate(order="original", limit=1)

# train from scratch (seed > 0)
# only use 10_000 mc samples (trainer_id = '132c5a41de356eda4032103ef56e8126')
# only train on the 1 neuron dataset (dl_id = '592885da0624c8a8c3073ec47d9bcfba')
# only transfer the 1 neuron task 1 dataset (orig_dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6')
restrictions = (
    "seed > 0 and "
    "trainer_id = '132c5a41de356eda4032103ef56e8126' and "
    "dl_id = '592885da0624c8a8c3073ec47d9bcfba' and "
    "orig_dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6'"
)

AdaptPriorResult.populate(
    restrictions,
    reserve_jobs=True,
    order="random",
    suppress_errors=True,
)
