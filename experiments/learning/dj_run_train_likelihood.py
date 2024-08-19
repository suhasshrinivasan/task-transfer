from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.result_tables import LikelihoodResult
from experiments.dj.trainer_tables import LLTrainerConfig
from task_transfer.utils.utils import dict_product

likelihood_configs = OrderedDict(
    seed=[42, 100],
    cond_dist=["indep_normal"],
    nonneg_transform=["exp"],
    n_layers=[1],
    nonlin=["none"],
    dropout_rate=[0.0],
    init_std=[1e-3],
    kwargs=[
        {
            "clamp_pre_scale": True,
            "pre_scale_max": 10.0,
        }
    ],
)

likelihood_configs_list = dict_product(likelihood_configs, insert_hash=True)

LikelihoodConfig.insert(likelihood_configs_list, skip_duplicates=True)

trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[250],
    batch_size=[128],
    early_stopping_threshold=[10],
    early_stopping_patience=[10],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

LLTrainerConfig.insert(trainer_configs_list, skip_duplicates=True)


# LikelihoodResult.populate(reserve_jobs=True, suppress_errors=True, order="random")


# learn likelihood model on flat haefner dataset
# but don't train all combinations
# pick the best likelihood model on hierarchical dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1
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

# grab ll_id from the best model
# set dl_id to the flat haefner dataset
restriction = (
    f"ll_id = '{best_val_likelihood_results['ll_id']}' "
    f"and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d'"
)
LikelihoodResult.FORCE_GPU = True
LikelihoodResult.populate(
    restriction,
    order="original",
    suppress_errors=True,
)
