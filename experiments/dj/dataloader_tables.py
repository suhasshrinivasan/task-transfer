import datajoint as dj

from task_transfer.ml_lib.data_loading import build_dataloaders_from_samples

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


@schema
class AltDataLoaderConfig(dj.Manual):
    """
    Dataloader configuration table for alternate data (used for transfer and adaptation experiments)
    """

    definition = """
    id: char(32)
    ---
    data_fname: varchar(255)
    train_prop: float
    val_prop: float
    """
