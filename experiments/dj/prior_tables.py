import datajoint as dj

from .schema import schema


@schema
class FlowPriorConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    seed: int
    flow_depth: int
    flow_initial_nonlin: varchar(32)    # flow direction is data -> latent
    flow_final_nonlin: varchar(32)
    flow_nonlin: varchar(32)
    flow_base_dist: varchar(32)
    affine_type: varchar(32)
    """


@schema
class AdaptPriorConfig(dj.Manual):
    """
    Config to adapt a prior to a new task
    At the moment, this indexes into FlowPriorResult and is used
    to adapt the prior model already trained and present
    in the FlowPriorResult table
    The adaptation mechanism is based on
    maximizing the evidence of observations in the new task
    and hence adapt prior additionally needs a likelihood model
    and thus indexes into LLResult table
    """

    definition = """
    seed: int   # negative seeds are used for training loaded models from scratch
    prior_fp_id: char(32) # to index into FlowPriorConfig
    prior_trainer_id: char(32)   # to index into FPTrainerConfig 
    likelihood_id: char(32) # to index into LikelihoodConfig
    likelihood_trainer_id: char(32)  # to index into LLTrainerConfig
    orig_dl_id: char(32) # to index into DataLoaderConfig used for the prior and likelihood training
    """
