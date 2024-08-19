from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.prior_tables import FlowPriorConfig
from experiments.dj.result_tables import FlowPriorResult
from experiments.dj.trainer_tables import FPTrainerConfig

# learn flow model on 1neuro haefner dataset
# but don't train all combinations
# pick the best flow model on hierarchical dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1
prior_config_proj_col = "fp_id"
# get a prior model that is fit on task 1 with original 45 neuron dataset
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
# set dl_id to the 1neuron haefner dataset
restriction = (
    f"fp_id = '{best_val_prior_results['fp_id']}' "
    f"and dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6'"
)
FlowPriorResult.populate(
    restriction,
    order="original",
    suppress_errors=True,
)
