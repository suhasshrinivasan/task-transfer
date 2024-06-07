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
