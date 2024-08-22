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

dataset_restriction = "dl_id = 'f7b32dd97feda9f34e2b47e24fa3d18b'"
best_val_prior_results = (FlowPriorResult & dataset_restriction).fetch(
    download_path=download_path, order_by="val_ll_mean DESC", as_dict=True, limit=k
)[0]

best_val_likelihood_results = (LikelihoodResult & dataset_restriction).fetch(
    download_path=download_path, order_by="val_ll_mean DESC", as_dict=True, limit=k
)[0]

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
    ],
    skip_duplicates=True,
)


trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[500],
    batch_size=[128],
    early_stopping_threshold=[1000],
    early_stopping_patience=[1000],
    mc_sample_size=[10_000],
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
# or 20_000 (trainer_id = 'eabf636932f56d44dcddf7300cf67f63')
# or either with 500 epochs ()
# only train on the 1 neuron task 2 dataset (dl_id = '592885da0624c8a8c3073ec47d9bcfba')
# also on high delta 1 neuron task 2 dataset (dl_id = '94efb58694007205fac996d7963f88c5')
# only transfer the 1 neuron task 1 dataset (orig_dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6')
# or the 1 neuron high delta task 1 dataset (orig_dl_id = 'f7b32dd97feda9f34e2b47e24fa3d18b')
restrictions = (
    "seed > 0 and "
    "(trainer_id = 'eabf636932f56d44dcddf7300cf67f63' or trainer_id = '38da520d4873f6c53b3dcf33746e62ab') and "
    "dl_id = '94efb58694007205fac996d7963f88c5' and "
    "orig_dl_id = 'f7b32dd97feda9f34e2b47e24fa3d18b'"
)

AdaptPriorResult.populate(
    restrictions,
    reserve_jobs=True,
    order="random",
    suppress_errors=True,
)
