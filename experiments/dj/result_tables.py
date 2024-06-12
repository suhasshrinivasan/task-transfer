import pickle
import tempfile
from pathlib import Path

import datajoint as dj
import torch

from task_transfer.utils.utils import make_hash

from ..learning.train_flow_prior import train_flow_prior
from ..learning.train_likelihood import train_likelihood
from ..learning.train_posterior import train_sbvp
from .dataloader_tables import DataLoaderConfig
from .dj_helpers import fetch_prior_cond_samples_path
from .likelihood_tables import LikelihoodConfig
from .posterior_tables import SBVGPConfig
from .prior_tables import FlowPriorConfig
from .schema import schema
from .trainer_tables import FPTrainerConfig, LLTrainerConfig, SBVGPTrainerConfig


@schema
class FlowPriorResult(dj.Computed):
    definition = """
        -> FlowPriorConfig.proj(fp_id='id')
        -> FPTrainerConfig.proj(trainer_id='id')
        -> DataLoaderConfig.proj(dl_id='id')
        ---
        train_ll_mean: double    # mean per dimension, per sample, in nats
        train_ll_sem: double    # standard error of the mean
        val_ll_mean: double
        val_ll_sem: double
        test_ll_mean: double
        test_ll_sem: double
        tracker_output: attach@external
        eval_output: attach@external
        model: attach@external
        """

    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # Fetch and prepare args
        print("Fetching FlowPriorConfig arguments...")
        prior_args = (FlowPriorConfig & {"id": key["fp_id"]}).fetch1()
        prior_args.pop("id")
        print("FlowPriorConfig arguments:", prior_args)

        print("Fetching FPTrainerConfig arguments...")
        trainer_args = (FPTrainerConfig & {"id": key["trainer_id"]}).fetch1()
        trainer_args.pop("id")
        print("FPTrainerConfig arguments:", trainer_args)

        print("Fetching DataLoaderConfig arguments...")
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args.pop("id")
        print("DataLoaderConfig arguments:", data_loader_args)

        # Train the model and get results
        print("Training flow prior model...")
        (
            model,
            train_ll_mean,
            train_ll_sem,
            val_ll_mean,
            val_ll_sem,
            test_ll_mean,
            test_ll_sem,
            tracker_output,
            eval_output,
        ) = train_flow_prior(data_loader_args, prior_args, trainer_args)
        print("Model training completed.")

        # Save model and results
        model_fname = Path(
            f"/tmp/{key['fp_id']}_{key['trainer_id']}_{key['dl_id']}_model.pt"
        )
        torch.save(model, model_fname)
        print("Model saved to:", model_fname)

        tracker_output_fname = Path(
            f"/tmp/{key['fp_id']}_{key['trainer_id']}_{key['dl_id']}_tracker_output.pkl"
        )
        with open(tracker_output_fname, "wb") as f:
            pickle.dump(tracker_output, f)
        print("Tracker output saved to:", tracker_output_fname)

        eval_output_fname = Path(
            f"/tmp/{key['fp_id']}_{key['trainer_id']}_{key['dl_id']}_eval_output.pkl"
        )
        with open(eval_output_fname, "wb") as f:
            pickle.dump(eval_output, f)
        print("Evaluation output saved to:", eval_output_fname)

        # Insert results
        print("Inserting results into the database...")
        self.insert1(
            {
                **key,
                "train_ll_mean": train_ll_mean,
                "train_ll_sem": train_ll_sem,
                "val_ll_mean": val_ll_mean,
                "val_ll_sem": val_ll_sem,
                "test_ll_mean": test_ll_mean,
                "test_ll_sem": test_ll_sem,
                "tracker_output": tracker_output_fname,
                "eval_output": eval_output_fname,
                "model": model_fname,
            }
        )
        print("Results inserted.")

        # Clean up
        print("Cleaning up temporary files...")
        model_fname.unlink()
        tracker_output_fname.unlink()
        eval_output_fname.unlink()
        print("Temporary files removed.")

        print("FlowPriorResult.make() completed.")


@schema
class LikelihoodResult(dj.Computed):
    definition = """
        -> LikelihoodConfig.proj(ll_id='id')
        -> LLTrainerConfig.proj(trainer_id='id')
        -> DataLoaderConfig.proj(dl_id='id')
        ---
        train_ll_mean: double    # mean per dimension, per sample, in nats
        train_ll_sem: double    # standard error of the mean
        val_ll_mean: double
        val_ll_sem: double
        test_ll_mean: double
        test_ll_sem: double
        tracker_output: attach@external
        eval_output: attach@external
        model: attach@external
        """

    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # Fetch and prepare args
        print("Fetching Likelihood arguments...")
        likelihood_args = (LikelihoodConfig & {"id": key["ll_id"]}).fetch1()
        likelihood_args.pop("id")
        print("Likelihood arguments:", likelihood_args)

        print("Fetching LLTrainerConfig arguments...")
        trainer_args = (LLTrainerConfig & {"id": key["trainer_id"]}).fetch1()
        trainer_args.pop("id")
        print("LLTrainerConfig arguments:", trainer_args)

        print("Fetching DataLoaderConfig arguments...")
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args.pop("id")
        print("DataLoaderConfig arguments:", data_loader_args)

        # Train the model and get results
        print("Training flow prior model...")
        (
            model,
            train_ll_mean,
            train_ll_sem,
            val_ll_mean,
            val_ll_sem,
            test_ll_mean,
            test_ll_sem,
            tracker_output,
            eval_output,
        ) = train_likelihood(data_loader_args, likelihood_args, trainer_args)
        print("Model training completed.")

        # Save model and results
        model_fname = Path(
            f"/tmp/{key['ll_id']}_{key['trainer_id']}_{key['dl_id']}_model.pt"
        )
        torch.save(model, model_fname)
        print("Model saved to:", model_fname)

        tracker_output_fname = Path(
            f"/tmp/{key['ll_id']}_{key['trainer_id']}_{key['dl_id']}_tracker_output.pkl"
        )
        with open(tracker_output_fname, "wb") as f:
            pickle.dump(tracker_output, f)
        print("Tracker output saved to:", tracker_output_fname)

        eval_output_fname = Path(
            f"/tmp/{key['ll_id']}_{key['trainer_id']}_{key['dl_id']}_eval_output.pkl"
        )
        with open(eval_output_fname, "wb") as f:
            pickle.dump(eval_output, f)
        print("Evaluation output saved to:", eval_output_fname)

        # Insert results
        print("Inserting results into the database...")
        self.insert1(
            {
                **key,
                "train_ll_mean": train_ll_mean,
                "train_ll_sem": train_ll_sem,
                "val_ll_mean": val_ll_mean,
                "val_ll_sem": val_ll_sem,
                "test_ll_mean": test_ll_mean,
                "test_ll_sem": test_ll_sem,
                "tracker_output": tracker_output_fname,
                "eval_output": eval_output_fname,
                "model": model_fname,
            }
        )
        print("Results inserted.")

        # Clean up
        print("Cleaning up temporary files...")
        model_fname.unlink()
        tracker_output_fname.unlink()
        eval_output_fname.unlink()
        print("Temporary files removed.")

        print("LikelihoodResult.make() completed.")


@schema
class FPSamplesConfig(dj.Manual):
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
class FPSamples(dj.Computed):
    """
    Table for Flow Prior Samples
    """

    definition = """
    -> FPSamplesConfig
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
        model_path = (
            FlowPriorResult
            & {
                "fp_id": key["fp_id"],
                "trainer_id": key["trainer_id"],
                "dl_id": key["dl_id"],
            }
        ).fetch1(download_path="/tmp")["model"]

        # Load the model
        print("Loading model...")
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        print("Model loaded.")
        with torch.no_grad():
            # Generate samples
            print("Generating samples...")
            samples = model.sample((key["n_samples"],))
        print("Samples generated.")

        print("Saving samples...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save samples
            samples_fname = f"{tmp_dir}/prior_{make_hash(key)}.pt"
            torch.save(samples, samples_fname)
            key["samples"] = samples_fname
            self.insert1(key)

        print("Samples saved.")


@schema
class MLPCondSamplesConfig(dj.Manual):
    """
    Table for MLP Samples Configuration
    """

    definition = """
    ll_id: char(32)    # from LikelihoodResult
    """


@schema
class MLPCondSamples(dj.Computed):
    definition = """
    -> FPSamplesConfig
    -> MLPCondSamplesConfig
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
            prior_samples_path = (
                FPSamples
                & {
                    "fp_id": key["fp_id"],
                    "dl_id": key["dl_id"],
                    "trainer_id": key["trainer_id"],
                    "n_samples": key["n_samples"],
                    "seed": key["seed"],
                }
            ).fetch1(download_path=tmp_dir)["samples"]
            prior_samples = torch.load(prior_samples_path, map_location="cpu")

            # Now fetch the likelihood model
            print("Fetching likelihood model...")
            likelihood_model_path = (
                LikelihoodResult
                & {
                    "ll_id": key["ll_id"],
                    "dl_id": key["dl_id"],
                    "trainer_id": key["trainer_id"],
                }
            ).fetch1(download_path=tmp_dir)["model"]

            # Load the likelihood model
            print("Loading likelihood model...")
            likelihood_model = torch.load(likelihood_model_path, map_location="cpu")
            likelihood_model.eval()
            with torch.no_grad():
                # Generate conditional samples
                print("Generating conditional samples...")
                samples = likelihood_model.sample(cond=prior_samples)
            print("Samples generated.")

            print("Saving samples...")
            # Save samples
            samples_fname = f"{tmp_dir}/cond_samples_{make_hash(key)}.pt"
            torch.save(samples, samples_fname)

            key["samples"] = samples_fname
            self.insert1(key)


@schema
class SBVGPResult(dj.Computed):
    """
    Result table for the Sample Based Variational Gamma Posterior
    """

    USE_WANDB = False
    FORCE_GPU = False

    definition = """
    -> SBVGPConfig.proj(sbvp_id='id')
    -> SBVGPTrainerConfig.proj(trainer_id='id')
    -> FPSamplesConfig.proj(fp_samples_id='fp_id', data_seed='seed')
    -> MLPCondSamplesConfig.proj(mlpcond_samples_id='ll_id')
    ---
    train_ll_mean: double    # mean per dimension, per sample, in nats
    train_ll_sem: double    # standard error of the mean
    val_ll_mean: double
    val_ll_sem: double
    test_ll_mean: double
    test_ll_sem: double

    train_ll_mean_sample: double    # mean per dimension, per sample, in nats
    train_ll_sem_sample: double    # standard error of the mean
    val_ll_mean_sample: double
    val_ll_sem_sample: double
    test_ll_mean_sample: double
    test_ll_sem_sample: double

    tracker_output: attach@external
    eval_output: attach@external
    model: attach@external
    """

    def make(self, key):
        print("SBVGPResult.make() called...")
        print("Received key ->", key)
        # get posterior args
        posterior_args = (SBVGPConfig & {"id": key["sbvp_id"]}).fetch1()
        posterior_args["dist"] = "gamma"

        # get the FPSamples and MLPCondSamples keys
        FPSamples_key = (
            FPSamplesConfig
            & {
                "fp_id": key["fp_samples_id"],
                "dl_id": key["dl_id"],
                "trainer_id": key["trainer_id"],
                "seed": key["data_seed"],
                "n_samples": key["n_samples"],
            }
        ).fetch1()
        MLPCondSamples_key = (
            MLPCondSamplesConfig & {"ll_id": key["mlpcond_samples_id"]}
        ).fetch1()
        prior_samples_path, cond_samples_path = fetch_prior_cond_samples_path(
            FPSamples, FPSamples_key, MLPCondSamples, MLPCondSamples_key
        )

        # get data_loader args
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args["sampled_responses_path"] = prior_samples_path
        data_loader_args["sampled_obs_path"] = cond_samples_path
        data_loader_args["data_seed"] = key["data_seed"]

        # get trainer args
        trainer_args = (SBVGPTrainerConfig & {"id": key["trainer_id"]}).fetch1()

        if self.FORCE_GPU:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise ValueError("GPU not available.")
        else:
            device = torch.device("cpu")
        trainer_args["device"] = device

        # train the model
        (
            model,
            train_ll_mean,
            train_ll_sem,
            val_ll_mean,
            val_ll_sem,
            test_ll_mean,
            test_ll_sem,
            train_ll_mean_sample,
            train_ll_sem_sample,
            val_ll_mean_sample,
            val_ll_sem_sample,
            test_ll_mean_sample,
            test_ll_sem_sample,
            tracker_output,
            eval_output,
        ) = train_sbvp(data_loader_args, posterior_args, trainer_args, self.USE_WANDB)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save model
            model_fname = Path(tmp_dir) / f"{make_hash(key)}_model.pt"
            torch.save(model, model_fname)

            # save tracker output
            tracker_output_fname = (
                Path(tmp_dir) / f"{make_hash(key)}_tracker_output.pkl"
            )
            with open(tracker_output_fname, "wb") as f:
                pickle.dump(tracker_output, f)

            # save eval output
            eval_output_fname = Path(tmp_dir) / f"{make_hash(key)}_eval_output.pkl"
            with open(eval_output_fname, "wb") as f:
                pickle.dump(eval_output, f)

            # insert results
            self.insert1(
                {
                    **key,
                    "train_ll_mean": train_ll_mean,
                    "train_ll_sem": train_ll_sem,
                    "val_ll_mean": val_ll_mean,
                    "val_ll_sem": val_ll_sem,
                    "test_ll_mean": test_ll_mean,
                    "test_ll_sem": test_ll_sem,
                    "train_ll_mean_sample": train_ll_mean_sample,
                    "train_ll_sem_sample": train_ll_sem_sample,
                    "val_ll_mean_sample": val_ll_mean_sample,
                    "val_ll_sem_sample": val_ll_sem_sample,
                    "test_ll_mean_sample": test_ll_mean_sample,
                    "test_ll_sem_sample": test_ll_sem_sample,
                    "tracker_output": tracker_output_fname,
                    "eval_output": eval_output_fname,
                    "model": model_fname,
                }
            )
            print("Results inserted.")
