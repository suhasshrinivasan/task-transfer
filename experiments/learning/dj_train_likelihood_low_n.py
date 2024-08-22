from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.dj_helpers import fetch_best_model_results
from experiments.dj.likelihood_tables import LikelihoodConfig
from experiments.dj.result_tables import LikelihoodResult
from experiments.dj.trainer_tables import LLTrainerConfig

# learn flow model on <4neuro datasets
# ids are
#   05977a317062b759857ee411a2e60648 (1 neuron task 1)
#   05977a317062b759857ee411a2e60648 (2 neurons task 1)
#   4477b5e82704db0bc19727864c7ef5aa (4 neurons task 1)
# but don't train all combinations
# pick the best likelihood model on hierarchical 45 neuron dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
# id for 45 neuron dataset is 260a5ea8175f75eaef132f42873ad14a

download_path = "/tmp"
criterion = "val_ll_mean"
k = 1
likelihood_config_proj_col = "ll_id"
dataset_restriction = "dl_id = '260a5ea8175f75eaef132f42873ad14a'"

best_val_likelihood_results = (LikelihoodResult & dataset_restriction).fetch(
    download_path=download_path, order_by="val_ll_mean DESC", as_dict=True, limit=k
)[0]

# best_val_likelihood_results = fetch_best_model_results(
#     result_table=LikelihoodResult,
#     config_table=LikelihoodConfig,
#     data_loader_config_table=data_loader_config_table,
#     trainer_config_table=LLTrainerConfig,
#     config_proj_col=likelihood_config_proj_col,
#     criterion=criterion,
#     k=k,
#     download_path=download_path,
# )

# grab ll_id from the best model
# set dl_id to the flat haefner dataset
restriction = (
    f"ll_id = '{best_val_likelihood_results['ll_id']}' "
    f"and dl_id = 'f7b32dd97feda9f34e2b47e24fa3d18b'"
)
LikelihoodResult.FORCE_GPU = True
# LikelihoodResult.populate(
#     restriction,
#     order="original",
#     suppress_errors=True,
# )
LikelihoodResult.populate(
    restriction,
    order="random",
    suppress_errors=True,
    reserve_jobs=True,
)
