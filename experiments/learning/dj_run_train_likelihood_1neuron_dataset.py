from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.result_tables import LikelihoodResult
from experiments.dj.trainer_tables import LLTrainerConfig

# learn flow model on 1neuro haefner dataset
# but don't train all combinations
# pick the best likelihood model on hierarchical 45 neuron dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
download_path = "/tmp"
criterion = "val_ll_mean"
k = 1
likelihood_config_proj_col = "ll_id"
dataset_restriction = "id = '260a5ea8175f75eaef132f42873ad14a'"
data_loader_config_table = DataLoaderConfig & dataset_restriction
best_val_likelihood_results = fetch_best_model_results(
    result_table=LikelihoodResult,
    config_table=LikelihoodConfig,
    data_loader_config_table=data_loader_config_table,
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
    f"and dl_id = 'f1ae78885d2ace1ba976199d4cf1a4d6'"
)
LikelihoodResult.FORCE_GPU = True
LikelihoodResult.populate(
    restriction,
    order="original",
    suppress_errors=True,
)
