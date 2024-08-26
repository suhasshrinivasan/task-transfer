import itertools as it
from collections import OrderedDict

from experiments.dj.dataloader_tables import DataLoaderConfig
from experiments.dj.result_tables import SIResult
from experiments.dj.sysident_tables import SIConfig
from experiments.dj.trainer_tables import SITrainerConfig
from task_transfer.utils.utils import dict_product

# train for 500 epochs
trainer_configs = OrderedDict(
    lr=[1e-3],
    weight_decay=[1e-3],
    n_epochs=[500],
    batch_size=[128],
    early_stopping_threshold=[10],
    early_stopping_patience=[10],
)

trainer_configs_list = dict_product(trainer_configs, insert_hash=True)

SITrainerConfig.insert(trainer_configs_list, skip_duplicates=True)

# learn gamma sysident model on 1neuro haefner dataset
# but don't train all combinations
# pick the sysident flow model on hierarchical dataset and borrow
# the same architecture and other hyperparameters
# write these as restrictions in the populate call
dataset_restriction = "dl_id = '260a5ea8175f75eaef132f42873ad14a'"
# the best sbvgp model from the 45 neuron case is in the DEPRECATED_SBVGPResult table
best_sysident_results = (SIResult & dataset_restriction).fetch(
    download_path="/tmp", order_by="val_ll_mean DESC", as_dict=True, limit=1
)
# grab si_id from the best model
# also grab the trainer_id
si_id = best_sysident_results[0]["si_id"]
trainer_id = best_sysident_results[0]["trainer_id"]

# set dl_id to the <=4neuron haefner dataset
sysident_restrictions = (
    f"si_id = '{si_id}' "
    f"and (dl_id = '8e9be142eedb21007255e89dbff362da' or dl_id = 'bb9bdd1ccd59e5a8c801d7f2d43e0317' or dl_id = 'd74090584b0b974c4444a5ec64c3d87d' or dl_id = '5352c4a57ef18797b082283de593157b')"
)
SIResult.populate(
    sysident_restrictions, reserve_jobs=True, suppress_errors=True, order="random"
)
# SIResult.populate(order="original", limit=1)
