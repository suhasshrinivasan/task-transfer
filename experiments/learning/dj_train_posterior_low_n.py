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
    MLPCondSamples2,
    MLPCondSamplesConfig2,
)
from experiments.dj.result_tables import SBVGPResult as DEPRECATED_SBVGPResult
from experiments.dj.result_tables import SBVGPResult2
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

dataset_restriction = "dl_id = 'bb9bdd1ccd59e5a8c801d7f2d43e0317'"
best_val_prior_results = (FlowPriorResult & dataset_restriction).fetch(
    download_path=download_path, order_by=f"{criterion} DESC", as_dict=True, limit=k
)[0]

best_val_likelihood_results = (LikelihoodResult & dataset_restriction).fetch(
    download_path=download_path, order_by=f"{criterion} DESC", as_dict=True, limit=k
)[0]

# use best prior and likelihood model result ids to populate the FP and MLPCond tables
fp_samples_configs = OrderedDict(
    fp_id=[best_val_prior_results["fp_id"]],
    dl_id=[best_val_prior_results["dl_id"]],
    trainer_id=[best_val_prior_results["trainer_id"]],
    n_samples=[10_000, 20_000, 30_000, 50_000],
    seed=[42],
)

fp_samples_configs_list = dict_product(fp_samples_configs, insert_hash=False)

FPSamplesConfig.insert(fp_samples_configs_list, skip_duplicates=True)

# # restrict to storing 10k samples for the 1 neuron case
# fp_samples_restrictions = (
#     "n_samples = 10000 and dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6'"
# )
# restrict to storing 50k samples for the 1 neuron case
fp_samples_restrictions = "dl_id = 'bb9bdd1ccd59e5a8c801d7f2d43e0317'"

FPSamples.populate(
    fp_samples_restrictions, reserve_jobs=True, suppress_errors=True, order="random"
)

mlpcond_samples_configs = OrderedDict(
    ll_id=[best_val_likelihood_results["ll_id"]],
    ll_trainer_id=[best_val_likelihood_results["trainer_id"]],
)
mlpcond_samples_configs_list = dict_product(mlpcond_samples_configs, insert_hash=False)

# Use MLPCondSamplesConfig2 instead of MLPCondSamplesConfig
MLPCondSamplesConfig2.insert(mlpcond_samples_configs_list, skip_duplicates=True)

# Use MLPCondSamples2 instead of MLPCondSamples
mlpcond_samples_restrictions = fp_samples_restrictions
MLPCondSamples2.populate(
    mlpcond_samples_restrictions,
    reserve_jobs=True,
    suppress_errors=True,
    order="random",
)

# do not populate all combinations
# get the posterior model that is fit on task 1 with original 45 neuron dataset
dataset_restriction = "dl_id = '260a5ea8175f75eaef132f42873ad14a'"
# the best sbvgp model from the 45 neuron case is in the DEPRECATED_SBVGPResult table
best_sbvgp_results = (DEPRECATED_SBVGPResult & dataset_restriction).fetch(
    download_path="/tmp", order_by="val_ll_mean DESC", as_dict=True, limit=1
)
# grab sbvp_id from the best model
# also grab the sbvp_trainer_id
sbvp_id = best_sbvgp_results[0]["sbvp_id"]
sbvp_trainer_id = best_sbvgp_results[0]["sbvp_trainer_id"]

# set dl_id to the 1neuron haefner dataset
sbvgp_restrictions = (
    f"sbvp_id = '{sbvp_id}' and dl_id = 'bb9bdd1ccd59e5a8c801d7f2d43e0317'"
)


# Use SBVGPResult2 instead of SBVGPResult
SBVGPResult2.USE_WANDB = False
SBVGPResult2.FORCE_GPU = True
# Use SBVGPResult2 instead of SBVGPResult
SBVGPResult2.populate(
    sbvgp_restrictions, reserve_jobs=True, suppress_errors=True, order="random"
)

# DEBUG ONLY
# SBVGPResult2.populate(
#     sbvgp_restrictions,
#     reserve_jobs=False,
#     suppress_errors=False,
#     order="original",
#     limit=1,
# )
