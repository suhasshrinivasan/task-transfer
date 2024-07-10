import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import FlowPriorResult
from experiments.dj.trainer_tables import FPTrainerConfig
from task_transfer.utils.utils import make_hash


def is_exceptional_flow_config(config):
    exceptional = False
    if "flow_depth" not in config:
        print(config)

    # Check for specific non-exceptional configuration
    if (
        config["flow_depth"] == 0
        and config["flow_initial_nonlin"] in {"log", "softplus"}
        and config["flow_nonlin"] == "none"
        and config["affine_type"] == "none"
        and "multivariate_normal" in config["flow_base_dist"]
        and "_lrmn" in config["flow_base_dist"]
    ):
        exceptional = False
    else:
        # Check for general exceptional conditions
        if (
            config["flow_depth"] == 0
            or config["flow_initial_nonlin"] == "log"
            or config["flow_nonlin"] == "none"
            or config["affine_type"] == "none"
            or "_lrmn" in config["flow_base_dist"]
        ):
            exceptional = True

    return exceptional


flow_prior_configs = OrderedDict(
    seed=[42, 100],
    flow_depth=[0, 2, 3],
    flow_initial_nonlin=["inv_softplus", "log"],
    flow_final_nonlin=["none"],
    flow_nonlin=["tanh", "leaky_relu", "none"],
    flow_base_dist=[
        "normal",
        "multivariate_normal",
        "multivariate_normal_lrmn",
        "lowrank_multivariate_normal_1",
        "lowrank_multivariate_normal_2",
        "lowrank_multivariate_normal_10",
    ],
    affine_type=["factorized", "full", "none"],
)

flow_prior_configs_list = []
for values in it.product(*flow_prior_configs.values()):
    config = {key: value for key, value in zip(flow_prior_configs.keys(), values)}
    if is_exceptional_flow_config(config):
        continue
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
    # add exceptions to handle custom combinations
    config = {key: value for key, value in zip(trainer_configs.keys(), values)}
    config["id"] = make_hash(config)
    trainer_configs_list.append(config)

FPTrainerConfig.insert(trainer_configs_list, skip_duplicates=True)

dataloader_configs = OrderedDict(
    data_fname=[
        "/src/project/data/synthetic/haefner_2afc/original_haefner_2afc_task_1_dataset.pkl",
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

# run only for the log initial nonlinearity
restriction_ids = (FlowPriorConfig & "flow_initial_nonlin = 'log'").fetch("id")
restrictions = f"fp_id = '{restriction_ids[0]}'"
for id in restriction_ids[1:]:
    restrictions += f" or fp_id = '{id}'"

FlowPriorResult.populate(
    restrictions,
    order="original",
    suppress_errors=True,
)

# FlowPriorResult.populate(
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )
