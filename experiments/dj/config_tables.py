import datajoint as dj
from schema import schema


@schema
class FlowPriorConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    dims: int
    flow_depth: int
    flow_initial_nonlin: varchar(32)    # flow direction is data -> latent
    flow_final_nonlin: varchar(32)
    flow_nonlin: varchar(32)
    flow_base_dist: varchar(32)
    affine_type: varchar(32)
    """


@schema
class LikelihoodConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    cond_dist: varchar(32)
    nonneg_transform: varchar(32)
    in_features: int
    out_features_core: int
    out_features_loc: int
    out_features_scale: int
    n_layers: int
    nonlin: varchar(32)
    dropout_rate: float
    init_std: float
    kwargs: longblob # other parameters such as pre_scale_max
    """


@schema
class TrainerConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    lr: float
    weight_decay: float
    n_epochs: int
    batch_size: int
    early_stopping_threshold: int
    early_stopping_patience: int
    logging_type: varchar(32)
    train_prop: float
    val_prop: float
    eval_criterion: varchar(32)
    eval_interval: int
    eval_params: longblob
    """


@schema
class DataLoaderConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    data_fname: varchar(255)
    train_prop: float
    val_prop: float
    """
