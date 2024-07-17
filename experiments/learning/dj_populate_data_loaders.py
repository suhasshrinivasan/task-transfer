import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import AltDataLoaderConfig, DataLoaderConfig
from task_transfer.utils.utils import dict_product

dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl",
        "/src/project/data/synthetic/haefner_2afc/flat_haefner_dataset.pkl",
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)


dataloader_configs_list = dict_product(dataloader_configs, insert_hash=True)

DataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)


dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_2_dataset.pkl",
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl",
        "/src/project/data/synthetic/haefner_2afc/flat_haefner_dataset.pkl",
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)

dataloader_configs_list = dict_product(dataloader_configs, insert_hash=True)

AltDataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)
