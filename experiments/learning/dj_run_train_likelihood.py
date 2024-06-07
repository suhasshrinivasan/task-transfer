import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
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


dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl"
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)

dataloader_configs_list = dict_product(dataloader_configs, insert_hash=True)

DataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)

LikelihoodResult.populate(reserve_jobs=True, suppress_errors=True, order="random")
