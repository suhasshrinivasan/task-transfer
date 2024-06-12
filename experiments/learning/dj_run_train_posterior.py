from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.posterior_tables import SBVGPConfig
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import (
    FlowPriorResult,
    FPSamples,
    FPSamplesConfig,
    LikelihoodResult,
    MLPCondSamples,
    MLPCondSamplesConfig,
    SBVGPResult,
)
from experiments.dj.trainer_tables import (
    FPTrainerConfig,
    LLTrainerConfig,
    SBVGPTrainerConfig,
)
from task_transfer.utils.utils import dict_product

# first extract best prior and likelihood models
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1

prior_config_proj_col = "fp_id"
best_val_prior_results = fetch_best_model_results(
    result_table=FlowPriorResult,
    config_table=FlowPriorConfig,
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

# use best prior and likelihood model result ids to populate the FP and MLPCond tables
fp_samples_configs = OrderedDict(
    fp_id=[best_val_prior_results["fp_id"]],
    dl_id=[best_val_prior_results["dl_id"]],
    trainer_id=[best_val_prior_results["trainer_id"]],
    n_samples=[10_000],
    seed=[42],
)

fp_samples_configs_list = dict_product(fp_samples_configs, insert_hash=False)

FPSamplesConfig.insert(fp_samples_configs_list, skip_duplicates=True)
FPSamples.populate(reserve_jobs=True, suppress_errors=True, order="random")

mlpcond_samples_configs = OrderedDict(
    ll_id=[best_val_likelihood_results["ll_id"]],
)
mlpcond_samples_configs_list = dict_product(mlpcond_samples_configs, insert_hash=False)

MLPCondSamplesConfig.insert(mlpcond_samples_configs_list, skip_duplicates=True)
MLPCondSamples.populate(reserve_jobs=True, suppress_errors=True, order="random")


posterior_configs = OrderedDict(
    seed=[42, 100],
    nonneg_transform=["exp"],
    n_layers=[1, 2, 3],
    nonlin=["none", "relu"],
    dropout_rate=[0.0],
    init_std=[1e-3],
    kwargs=[
        {
            "clamp_pre_conc": True,
            "pre_conc_max": 4.0,
            "clamp_pre_rate": True,
            "pre_rate_min": -1.6,
        }
    ],
)

posterior_configs_list = dict_product(posterior_configs, insert_hash=True)

SBVGPConfig.insert(posterior_configs_list, skip_duplicates=True)

trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[2],
    batch_size=[128],
    early_stopping_threshold=[10],
    early_stopping_patience=[10],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

SBVGPTrainerConfig.insert(trainer_configs_list, skip_duplicates=True)
SBVGPResult.USE_WANDB = True
SBVGPResult.FORCE_GPU = True
SBVGPResult.populate(reserve_jobs=True, suppress_errors=True, order="random")
# SBVGPResult.populate(order="original", limit=1)
