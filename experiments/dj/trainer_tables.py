import datajoint as dj

from .schema import schema


@schema
class FPTrainerConfig(dj.Manual):
    """
    Flow prior trainer configuration table
    """

    definition = """
    id: char(32)
    ---
    lr: float
    weight_decay: float
    n_epochs: int 
    batch_size: int
    early_stopping_threshold: int
    early_stopping_patience: int
    """


@schema
class LLTrainerConfig(dj.Manual):
    """
    Likelihood trainer configuration table
    """

    definition = """
    id: char(32)
    ---
    lr: float
    weight_decay: float
    n_epochs: int
    batch_size: int
    early_stopping_threshold: int
    early_stopping_patience: int
    """


@schema
class SBVGPTrainerConfig(dj.Manual):
    """
    Sample Based Variational Gamma Posterior trainer configuration table
    """

    definition = """
    id: char(32)
    ---
    lr: float
    weight_decay: float
    n_epochs: int
    batch_size: int
    early_stopping_threshold: int
    early_stopping_patience: int
    """
