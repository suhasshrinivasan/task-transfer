import datajoint as dj

from .schema import schema


@schema
class DataLoaderConfig(dj.Manual):
    """
    Dataloader configuration table
    """

    definition = """
    id: char(32)
    ---
    data_fname: varchar(255)
    train_prop: float
    val_prop: float
    """
