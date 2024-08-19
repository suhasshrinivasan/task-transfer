import datajoint as dj

from .schema import schema


@schema
class SBVGPConfig(dj.Manual):
    """
    Sample Based Variational Gamma Posterior configuration table
    """

    definition = """
    id: char(32)
    ---
    seed: int
    nonneg_transform: varchar(32)
    n_layers: int
    nonlin: varchar(32)
    dropout_rate: float
    init_std: float
    kwargs: longblob
    """


@schema
class VPostPriorConfig(dj.Manual):
    """
    Variational Posterior and Prior configuration table
    """

    definition = """
    id: char(32)
    ---
    seed: int
    post_dist_type: varchar(50)
    post_nonneg_transform: varchar(32)
    post_n_layers: int
    post_nonlin: varchar(32)
    post_dropout_rate: float
    post_init_std: float
    post_kwargs: longblob
    prior_fp_id: char(32) # to index into FlowPriorConfig
    prior_trainer_id: char(32)   # to index into FPTrainerConfig 
    likelihood_id: char(32) # to index into LikelihoodConfig
    likelihood_trainer_id: char(32)  # to index into LLTrainerConfig
    orig_dl_id: char(32) # to index into DataLoaderConfig used to id the prior and likelihood
"""
