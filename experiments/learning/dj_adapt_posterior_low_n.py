from collections import OrderedDict

from experiments.dj.dataloader_tables import AltDataLoaderConfig, DataLoaderConfig
from experiments.dj.posterior_tables import SBVGPConfig
from experiments.dj.result_tables import (
    AdaptMLPCondSamples,
    AdaptPriorResult,
    AdaptPriorSamples,
    AdaptPriorSamplesConfig,
    SBVGPAdaptedResult,
)
from experiments.dj.result_tables import SBVGPResult as DEPRECATED_SBVGPResult
from task_transfer.utils.utils import dict_product

# first extract best prior and likelihood models
download_path = "/tmp"
criterion = "val_marginal_obs_ll_mean"
k = 1

dataset_restriction = "dl_id = '9ef3ae6fea33eba634d928a88b866836'"
best_val_adapt_prior = (AdaptPriorResult & dataset_restriction).fetch(
    download_path=download_path, order_by=f"{criterion} DESC", as_dict=True, limit=k
)[0]


# use best prior and likelihood model result ids to populate the FP and MLPCond tables
fp_samples_configs = OrderedDict(
    adapt_prior_seed=[best_val_adapt_prior["seed"]],
    prior_fp_id=[best_val_adapt_prior["prior_fp_id"]],
    prior_trainer_id=[best_val_adapt_prior["prior_trainer_id"]],
    likelihood_id=[best_val_adapt_prior["likelihood_id"]],
    likelihood_trainer_id=[best_val_adapt_prior["likelihood_trainer_id"]],
    orig_dl_id=[best_val_adapt_prior["orig_dl_id"]],
    adapt_prior_trainer_id=[best_val_adapt_prior["trainer_id"]],
    alt_dl_id=[best_val_adapt_prior["dl_id"]],
    n_samples=[10_000, 20_000, 30_000, 50_000],
    seed=[42, 100],
)

fp_samples_configs_list = dict_product(fp_samples_configs, insert_hash=False)

AdaptPriorSamplesConfig.insert(fp_samples_configs_list, skip_duplicates=True)

AdaptPriorSamples.populate(reserve_jobs=True, suppress_errors=True, order="random")
AdaptMLPCondSamples.populate(reserve_jobs=True, suppress_errors=True, order="random")

# On samples above, train sample-based variational gamma posterior models
# but not all models, pick the best one from old 45-neuron case and use the same architecture
# to train the new model
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
    f"sbvp_id = '{sbvp_id}' "
    # f"and sbvp_trainer_id = '{sbvp_trainer_id}' "
    f"and alt_dl_id = '9ef3ae6fea33eba634d928a88b866836'"
)

SBVGPAdaptedResult.populate(
    sbvgp_restrictions, reserve_jobs=True, suppress_errors=True, order="random"
)
