import datajoint as dj

from .schema import schema


@schema
class LikelihoodConfig(dj.Manual):
    definition = """
    id: char(32)
    ---
    seed: int
    cond_dist: varchar(32)
    nonneg_transform: varchar(32)
    n_layers: int
    nonlin: varchar(32)
    dropout_rate: float
    init_std: float
    kwargs: longblob # other parameters such as pre_scale_max
    """
