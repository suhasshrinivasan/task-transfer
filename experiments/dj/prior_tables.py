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
