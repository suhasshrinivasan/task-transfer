from collections import OrderedDict

from experiments.dj.dataloader_tables import AltDataLoaderConfig
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.posterior_tables import SBVGPConfig, VPostPriorConfig
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import (
    FlowPriorResult,
    FPSamples,
    FPSamplesConfig,
    LikelihoodResult,
    MLPCondSamples,
    MLPCondSamplesConfig,
    SBVGPResult,
    VPostPriorResult,
)
from experiments.dj.trainer_tables import (
    FPTrainerConfig,
    LLTrainerConfig,
    SBVGPTrainerConfig,
    VPTrainerConfig,
)
from task_transfer.utils.utils import dict_product

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

vpostprior_configs = dict_product(
    OrderedDict(
        seed=[42, 100, -42, -100],
        post_dist_type=["gamma"],
        post_nonneg_transform=["exp"],
        post_n_layers=[1, 2, 3, 4, 5],
        post_nonlin=["relu"],
        post_dropout_rate=[0.0],
        post_init_std=[0.001],
        post_kwargs=[
            {
                "clamp_pre_conc": True,
                "pre_conc_max": 4.0,
                "clamp_pre_rate": True,
                "pre_rate_min": -1.6,
            }
        ],
        prior_fp_id=[best_val_prior_results["fp_id"]],
        prior_trainer_id=[best_val_prior_results["trainer_id"]],
        likelihood_id=[best_val_likelihood_results["ll_id"]],
        likelihood_trainer_id=[best_val_likelihood_results["trainer_id"]],
        orig_dl_id=[best_val_likelihood_results["dl_id"]],
    )
)


VPostPriorConfig.insert(vpostprior_configs, skip_duplicates=True)

vptrainer_configs = dict_product(
    OrderedDict(
        n_bound_samples=[1, 100, 1000],
        bound_type=["elbo", "iw"],
        lr=[1e-3],
        weight_decay=[1e-5, 1e-3, 1e-1],
        n_epochs=[100, 300],
        batch_size=[128],
        early_stopping_threshold=[10],
        early_stopping_patience=[10],
    )
)

VPTrainerConfig.insert(vptrainer_configs, skip_duplicates=True)

restrictions = f"dl_id = '3d740ef65d4ec3d651cb862eb90143df'"

VPostPriorResult.FORCE_GPU = True
# VPostPriorResult.populate(
#     restrictions,
#     order="original",
#     suppress_errors=True,
# )
VPostPriorResult.populate(
    restrictions,
    order="random",
    suppress_errors=True,
    reserve_jobs=True,
)
