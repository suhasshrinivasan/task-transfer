import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import FlowPriorResult
from experiments.dj.trainer_tables import FPTrainerConfig
from task_transfer.utils.utils import make_hash


def is_exceptional_flow_config(config):
    exceptional = False

    # Check for specific non-exceptional configuration
    # These are
    # 1. log-normal distribution
    # 2. exponential distribution
    if (
        config["flow_depth"] == 0
        and config["flow_initial_nonlin"] in {"log", "softplus"}
        and config["flow_nonlin"] == "none"
        and config["affine_type"] == "none"
        and "multivariate_normal" in config["flow_base_dist"]
        and "_lrmn" in config["flow_base_dist"]
    ):
        exceptional = False
    elif (
        config["flow_depth"] == 0
        and config["flow_initial_nonlin"] == "none"
        and config["flow_nonlin"] == "none"
        and config["affine_type"] == "none"
        and "exponential" in config["flow_base_dist"]
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


# run only for the log initial nonlinearity
# restriction_ids = (FlowPriorConfig & "flow_initial_nonlin = 'log'").fetch("id")
# restrictions = f"fp_id = '{restriction_ids[0]}'"
# for id in restriction_ids[1:]:
#     restrictions += f" or fp_id = '{id}'"

# FlowPriorResult.populate(
#     restrictions,
#     order="original",
#     suppress_errors=True,
# )

# FlowPriorResult.populate(
#     reserve_jobs=True,
#     order="random",
#     suppress_errors=True,
# )

# learn flow model on flat haefner dataset
# but don't train all combinations
# pick the best flow model on hierarchical 45 neuron dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1
prior_config_proj_col = "fp_id"
# get a prior model that is fit on task 1 data
dataset_restriction = "id = '260a5ea8175f75eaef132f42873ad14a'"
data_loader_config_table = DataLoaderConfig & dataset_restriction
best_val_prior_results = fetch_best_model_results(
    result_table=FlowPriorResult,
    config_table=FlowPriorConfig,
    data_loader_config_table=data_loader_config_table,
    trainer_config_table=FPTrainerConfig,
    config_proj_col=prior_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)
# grab fp_id from the best model
# set dl_id to the flat haefner dataset
restriction = (
    f"fp_id = '{best_val_prior_results['fp_id']}' "
    f"and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d'"
)
FlowPriorResult.populate(
    restriction,
    order="original",
    suppress_errors=True,
)

# also learn flow model without correlation, since the data is flat anyway
no_corr_restriction = "flow_base_dist = 'normal' and affine_type = 'factorized'"
flow_prior_config_table = FlowPriorConfig & no_corr_restriction
best_val_prior_results = fetch_best_model_results(
    result_table=FlowPriorResult,
    config_table=flow_prior_config_table,
    data_loader_config_table=data_loader_config_table,
    trainer_config_table=FPTrainerConfig,
    config_proj_col=prior_config_proj_col,
    criterion=criterion,
    k=k,
    download_path=download_path,
)
restriction = (
    f"fp_id = '{best_val_prior_results['fp_id']}' "
    f"and dl_id = 'b8379e7d6998fc94a08a9a3742eec12d'"
)
FlowPriorResult.populate(
    restriction,
    order="original",
    suppress_errors=True,
)
