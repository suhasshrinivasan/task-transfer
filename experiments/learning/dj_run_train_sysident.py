import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.result_tables import SIResult
from experiments.dj.sysident_tables import SIConfig
from experiments.dj.trainer_tables import SITrainerConfig
from task_transfer.utils.utils import dict_product

sysident_configs = OrderedDict(
    seed=[42, 100],
    cond_dist=["gamma"],
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

sysident_configs_list = dict_product(sysident_configs, insert_hash=True)

SIConfig.insert(sysident_configs_list, skip_duplicates=True)

trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[250],
    batch_size=[128],
    early_stopping_threshold=[10],
    early_stopping_patience=[10],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

SITrainerConfig.insert(trainer_configs_list, skip_duplicates=True)


dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl"
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)

dataloader_configs_list = dict_product(dataloader_configs, insert_hash=True)

DataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)

SIResult.populate(reserve_jobs=True, suppress_errors=True, order="random")
# SIResult.populate(order="original", limit=1)
