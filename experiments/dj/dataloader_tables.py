import datajoint as dj
import torch
import tempfile

from .result_tables import FlowPriorResult, LikelihoodResult
from .schema import schema
from task_transfer.ml_lib.data_loading import build_dataloaders_from_samples

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
class FP_SamplesConfig(dj.Manual):
    """
    Table for Flow Prior Samples Configuration
    """

    definition = """
    fp_id: char(32)    # from FlowPriorResult
    dl_id: char(32)    # from FlowPriorResult and DataLoaderConfig
    trainer_id: char(32)    # from FlowPriorResult and FPTrainerConfig
    seed: int
    n_samples: int
    """


@schema
class FP_Samples(dj.Computed):
    """
    Table for Flow Prior Samples
    """

    definition = """
    -> FP_SamplesConfig
    ---
    samples: attach@external
    """

    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # set the seed
        torch.manual_seed(key["seed"])

        # Fetch and prepare args
        print("Fetching FlowPriorResult arguments...")
        model_path = (FlowPriorResult & {"id": key["fp_id"]}).fetch1(
            download_path="/tmp"
        )["model"]

        # Load the model
        print("Loading model...")
        model = torch.load(model_path)
        model.eval()
        print("Model loaded.")

        # Generate samples
        print("Generating samples...")
        samples = model.sample(key["n_samples"])
        print("Samples generated.")

        print("Saving samples...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save samples
            samples_fname = f"{tmp_dir}/prior_{key["fp_id"]}_{key["seed"]}_{key["n_samples"]}.pt"
            torch.save(samples, samples_fname)
            key["samples"] = samples_fname
            self.insert1(key)

        print("Samples saved.")


@schema
class MLPCond_SamplesConfig(dj.Manual):
    """
    Table for MLP Samples Configuration
    """

    definition = """
    likelihood_id: char(32)    # from LikelihoodResult
    """

@schema
class MLPCond_Samples(dj.Computed):
    definition = """
    -> FP_SamplesConfig
    -> MLPCond_SamplesConfig
    ---
    samples: attach@external
    """
    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # Set the seed
        torch.manual_seed(key["seed"])

        # First fetch prior samples
        print("Fetching prior samples...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            prior_samples_path = (FP_Samples & {"fp_id": key["fp_id"], "seed": key["seed"]}).fetch1(
                download_path=tmp_dir
            )["samples"]
            prior_samples = torch.load(prior_samples_path)

            # Now fetch the likelihood model
            print("Fetching likelihood model...")
            likelihood_model_path = (LikelihoodResult & {"id": key["likelihood_id"]}).fetch1(
                download_path=tmp_dir
            )["model"]

            # Load the likelihood model
            print("Loading likelihood model...")
            likelihood_model = torch.load(likelihood_model_path)

            # Generate conditional samples
            print("Generating conditional samples...")
            samples = likelihood_model.sample(cond=prior_samples)
            print("Samples generated.")

            print("Saving samples...")
            # Save samples
            samples_fname = f"{tmp_dir}/cond_samples_{key["likelihood_id"]}_{key["seed"]}.pt"
            torch.save(samples, samples_fname)

            key["samples"] = samples_fname
            self.insert1(key)


def fetch_samples_path_from_dj(FP_Samples_key, MLPCond_Samples_key):
    # Fetch the samples
    prior_samples_path = (FP_Samples & FP_Samples_key).fetch1(download_path="/tmp")["samples"]
    cond_samples_path = (MLPCond_Samples & MLPCond_Samples_key).fetch1(download_path="/tmp")["samples"]
    return prior_samples_path, cond_samples_path
