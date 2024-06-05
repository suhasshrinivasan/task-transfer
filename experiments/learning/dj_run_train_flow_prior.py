import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import FlowPriorResult
from experiments.dj.trainer_tables import FPTrainerConfig
from task_transfer.utils.utils import make_hash

flow_prior_configs = OrderedDict(
    seed=[42, 100],
    flow_depth=[2, 3],
    flow_initial_nonlin=["inv_softplus"],
    flow_final_nonlin=["none"],
    flow_nonlin=["tanh", "leaky_relu"],
    flow_base_dist=[
        "normal",
        "multivariate_normal",
        "lowrank_multivariate_normal_1",
        "lowrank_multivariate_normal_2",
        "lowrank_multivariate_normal_10",
    ],
    affine_type=["factorized", "full"],
)

flow_prior_configs_list = []
for values in it.product(*flow_prior_configs.values()):
    config = {key: value for key, value in zip(flow_prior_configs.keys(), values)}
    config["id"] = make_hash(config)
    flow_prior_configs_list.append(config)

FlowPriorConfig.insert(flow_prior_configs_list, skip_duplicates=True)

trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[250, 600],
    batch_size=[128],
    early_stopping_threshold=[10],
    early_stopping_patience=[10],
)

trainer_configs_list = []
for values in it.product(*trainer_configs.values()):
    config = {key: value for key, value in zip(trainer_configs.keys(), values)}
    config["id"] = make_hash(config)
    trainer_configs_list.append(config)

FPTrainerConfig.insert(trainer_configs_list, skip_duplicates=True)

dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl"
    ],
    train_prop=[0.7],
    val_prop=[0.2],
)

dataloader_configs_list = []
for values in it.product(*dataloader_configs.values()):
    config = {key: value for key, value in zip(dataloader_configs.keys(), values)}
    config["id"] = make_hash(config)
    dataloader_configs_list.append(config)

DataLoaderConfig.insert(dataloader_configs_list, skip_duplicates=True)

FlowPriorResult.populate(
    reserve_jobs=True,
    order="random",
    suppress_errors=True,
)
