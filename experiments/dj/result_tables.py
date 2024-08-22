import pickle
import tempfile
from pathlib import Path

import datajoint as dj
import torch

from task_transfer.utils.utils import make_hash

from ..learning.adapt_prior import adapt_prior
from ..learning.train_flow_prior import train_flow_prior
from ..learning.train_likelihood import train_likelihood
from ..learning.train_posterior import train_sbvp, train_vpost_prior
from ..learning.train_sysident import train_sysident
from .dataloader_tables import AltDataLoaderConfig, DataLoaderConfig
from .dj_helpers import fetch_prior_cond_samples_path
from .likelihood_tables import LikelihoodConfig
from .posterior_tables import SBVGPConfig, VPostPriorConfig
from .prior_tables import AdaptPriorConfig, FlowPriorConfig
from .schema import schema
from .sysident_tables import SIConfig
from .trainer_tables import (
    AdaptPriorTrainer,
    FPTrainerConfig,
    LLTrainerConfig,
    SBVGPTrainerConfig,
    SITrainerConfig,
    VPTrainerConfig,
)


@schema
class FlowPriorResult(dj.Computed):

    FORCE_GPU = False
    USE_WANDB = False

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

        if self.FORCE_GPU:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                raise ValueError("GPU not available.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer_args["device"] = device

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
        ) = train_flow_prior(
            data_loader_args, prior_args, trainer_args, self.USE_WANDB, dj.conn()
        )
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

    FORCE_GPU = False

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

        if self.FORCE_GPU:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise ValueError("GPU not available.")
        else:
            device = torch.device("cpu")
        trainer_args["device"] = device

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
class SIResult(dj.Computed):
    """
    System identification result table
    """

    USE_WANDB = False
    FORCE_GPU = False

    definition = """
        -> SIConfig.proj(si_id='id')
        -> SITrainerConfig.proj(trainer_id='id')
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
        print("Fetching SI arguments...")
        sysident_args = (SIConfig & {"id": key["si_id"]}).fetch1()
        sysident_args.pop("id")
        print("SI arguments:", sysident_args)

        print("Fetching SITrainerConfig arguments...")
        trainer_args = (SITrainerConfig & {"id": key["trainer_id"]}).fetch1()
        trainer_args.pop("id")
        print("SITrainer arguments:", trainer_args)

        print("Fetching DataLoaderConfig arguments...")
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args.pop("id")
        print("DataLoaderConfig arguments:", data_loader_args)

        if self.FORCE_GPU:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                raise ValueError("GPU not available.")
        else:
            device = torch.device("cpu")
        trainer_args["device"] = device

        # Train the model and get results
        print("Training sysident model...")
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
        ) = train_sysident(
            data_loader_args, sysident_args, trainer_args, self.USE_WANDB
        )
        print("Model training completed.")

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


# MLPCondSamplesConfig2 was created to replace MLPCondSamplesConfig
# MLPCondSamplesConfig2 includes the ll_trainer_id which is essential
# for identifying the likelihood model
# MLPCondSamplesConfig is DEPRECATED in favor of MLPCondSamplesConfig2
@schema
class MLPCondSamplesConfig2(dj.Manual):
    """
    Table for MLP Samples Configuration

    NOTE:
    MLPCondSamplesConfig2 was created to replace MLPCondSamplesConfig
    MLPCondSamplesConfig2 includes the ll_trainer_id which is essential
    for identifying the likelihood model
    MLPCondSamplesConfig is DEPRECATED in favor of MLPCondSamplesConfig2
    """

    definition = """
    ll_id: char(32)    # from LikelihoodResult
    ll_trainer_id: char(32)    # from LikelihoodResult and LLTrainerConfig
    """


# MLPCondSamples2 was created to replace MLPCondSamples
# MLPCondSamples2 uses the new MLPCondSamplesConfig2
@schema
class MLPCondSamples2(dj.Computed):
    """
    Table for MLP Conditional Samples with the new MLPCondSamplesConfig2

    NOTE:
    MLPCondSamples2 was created to replace MLPCondSamples
    MLPCondSamples2 uses the new MLPCondSamplesConfig2
    """

    definition = """
    -> FPSamplesConfig
    -> MLPCondSamplesConfig2
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
                    "trainer_id": key["ll_trainer_id"],
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


# SBVGPResult2 was created to replace SBVGPResult
# SBVGPResult2 uses the new MLPCondSamplesConfig2 and MLPCondSamples2
@schema
class SBVGPResult2(dj.Computed):
    """
    Result table for the Sample Based Variational Gamma Posterior with the new MLPCondSamplesConfig2

    NOTE:
    SBVGPResult2 was created to replace SBVGPResult
    SBVGPResult2 uses the new MLPCondSamplesConfig2 and MLPCondSamples2
    """

    USE_WANDB = False
    FORCE_GPU = False

    definition = """
    -> SBVGPConfig.proj(sbvp_id='id')
    -> SBVGPTrainerConfig.proj(sbvp_trainer_id='id')
    -> FPSamplesConfig.proj(fp_samples_id='fp_id', data_seed='seed')
    -> MLPCondSamplesConfig2.proj(mlpcond_samples_id='ll_id')
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
        # Use MLPCondSamplesConfig2 instead of MLPCondSamplesConfig
        MLPCondSamples_key = (
            MLPCondSamplesConfig2 & {"ll_id": key["mlpcond_samples_id"]}
        ).fetch1()
        # Use MLPCondSamples2 instead of MLPCondSamples
        prior_samples_path, cond_samples_path = fetch_prior_cond_samples_path(
            FPSamples, FPSamples_key, MLPCondSamples2, MLPCondSamples_key
        )

        # get data_loader args
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args["sampled_responses_path"] = prior_samples_path
        data_loader_args["sampled_obs_path"] = cond_samples_path
        data_loader_args["data_seed"] = key["data_seed"]

        # get trainer args
        trainer_args = (SBVGPTrainerConfig & {"id": key["sbvp_trainer_id"]}).fetch1()

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
            print("Results inserted.l")


# DEPRECATE THIS IN FAVOR OF MLPCondSamplesConfig2
# This table does not have the likelihood trainer id which
# is essential for identifying the likelihood model
@schema
class MLPCondSamplesConfig(dj.Manual):
    """
    Table for MLP Samples Configuration
    """

    definition = """
    ll_id: char(32)    # from LikelihoodResult
    """


# DEPRECATE THIS IN FAVOR OF MLPCondSamples2
# This table inherits from MLPCondSamplesConfig which is deprecated
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


# DEPRECATE THIS IN FAVOR OF SBVGPResult2
# This table inherits from MLPCondSamplesConfig which is deprecated
@schema
class SBVGPResult(dj.Computed):
    """
    Result table for the Sample Based Variational Gamma Posterior
    """

    USE_WANDB = False
    FORCE_GPU = False

    definition = """
    -> SBVGPConfig.proj(sbvp_id='id')
    -> SBVGPTrainerConfig.proj(sbvp_trainer_id='id')
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
        trainer_args = (SBVGPTrainerConfig & {"id": key["sbvp_trainer_id"]}).fetch1()

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


@schema
class AdaptPriorSamplesConfig(dj.Manual):
    """
    Table for Adapt Prior Samples Configuration
    """

    definition = """
    adapt_prior_seed: int   # seed for adapt prior model
    prior_fp_id: char(32) # to index into FlowPriorConfig
    prior_trainer_id: char(32)   # to index into FPTrainerConfig 
    likelihood_id: char(32) # to index into LikelihoodConfig
    likelihood_trainer_id: char(32)  # to index into LLTrainerConfig
    orig_dl_id: char(32) # to index into DataLoaderConfig used for the prior and likelihood training
    adapt_prior_trainer_id: char(32)  # to index into AdaptPriorTrainer
    alt_dl_id: char(32) # to index into AltDataLoaderConfig
    n_samples: int  # number of samples to generate
    seed: int  # seed for generating samples
    """


@schema
class AdaptPriorSamples(dj.Computed):
    """
    Table for Adapt Prior Samples
    """

    definition = """
    -> AdaptPriorSamplesConfig
    ---
    samples: attach@external
    """

    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # Set the seed
        torch.manual_seed(key["seed"])

        # Fetch and prepare args
        print("Fetching AdaptPriorResult model...")
        model_path = (
            AdaptPriorResult
            & {
                "seed": key["adapt_prior_seed"],
                "prior_fp_id": key["prior_fp_id"],
                "prior_trainer_id": key["prior_trainer_id"],
                "likelihood_id": key["likelihood_id"],
                "likelihood_trainer_id": key["likelihood_trainer_id"],
                "orig_dl_id": key["orig_dl_id"],
                "trainer_id": key["adapt_prior_trainer_id"],
                "dl_id": key["alt_dl_id"],
            }
        ).fetch1(download_path="/tmp")["model"]

        # Load the model
        print("Loading model...")
        model = torch.load(model_path, map_location="cpu")
        prior_model = model.prior
        prior_model.eval()
        print("Model loaded.")
        with torch.no_grad():
            # Generate samples
            print("Generating samples...")
            samples = prior_model.sample((key["n_samples"],))
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
class AdaptMLPCondSamples(dj.Computed):
    """
    Table for Adapt Prior Samples
    """

    definition = """
    -> AdaptPriorSamplesConfig
    ---
    samples: attach@external
    """

    def make(self, key):
        # Print the provided key
        print("Received key ->", key)

        # Set the seed
        torch.manual_seed(key["seed"])

        # Fetch and prepare args
        print("Fetching AdaptPriorResult model...")
        model_path = (
            AdaptPriorResult
            & {
                "seed": key["adapt_prior_seed"],
                "prior_fp_id": key["prior_fp_id"],
                "prior_trainer_id": key["prior_trainer_id"],
                "likelihood_id": key["likelihood_id"],
                "likelihood_trainer_id": key["likelihood_trainer_id"],
                "orig_dl_id": key["orig_dl_id"],
                "trainer_id": key["adapt_prior_trainer_id"],
                "dl_id": key["alt_dl_id"],
            }
        ).fetch1(download_path="/tmp")["model"]

        # Fetch prior samples
        print("Fetching prior samples...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            prior_samples_path = (
                AdaptPriorSamples
                & {
                    "adapt_prior_seed": key["adapt_prior_seed"],
                    "prior_fp_id": key["prior_fp_id"],
                    "prior_trainer_id": key["prior_trainer_id"],
                    "likelihood_id": key["likelihood_id"],
                    "likelihood_trainer_id": key["likelihood_trainer_id"],
                    "orig_dl_id": key["orig_dl_id"],
                    "adapt_prior_trainer_id": key["adapt_prior_trainer_id"],
                    "alt_dl_id": key["alt_dl_id"],
                    "n_samples": key["n_samples"],
                    "seed": key["seed"],
                }
            ).fetch1(download_path=tmp_dir)["samples"]
            prior_samples = torch.load(prior_samples_path, map_location="cpu")

            # Load the model
            print("Loading model...")
            model = torch.load(model_path, map_location="cpu")
            conditional_model = model.conditional
            conditional_model.eval()
            with torch.no_grad():
                # Generate conditional samples
                print("Generating conditional samples...")
                samples = conditional_model.sample(cond=prior_samples)
            print("Samples generated.")

            print("Saving samples...")
            # Save samples
            samples_fname = f"{tmp_dir}/cond_samples_{make_hash(key)}.pt"
            torch.save(samples, samples_fname)

            key["samples"] = samples_fname
            self.insert1(key)


# @schema
# class SBVGPAdaptedResult(dj.Computed):
#     """
#     Result table for the Sample Based Variational Gamma Posterior incorporating adapted prior table AdaptPriorResult
#     """

#     USE_WANDB = False
#     FORCE_GPU = False

#     definition = """
#     -> SBVGPConfig.proj(sbvp_id='id')
#     -> SBVGPTrainerConfig.proj(sbvp_trainer_id='id')
#     -> AdaptPriorSamplesConfig.proj(fp_samples_id='fp_id', data_seed='seed')
#     ---
#     train_ll_mean: double    # mean per dimension, per sample, in nats
#     train_ll_sem: double    # standard error of the mean
#     val_ll_mean: double
#     val_ll_sem: double
#     test_ll_mean: double
#     test_ll_sem: double

#     train_ll_mean_sample: double    # mean per dimension, per sample, in nats
#     train_ll_sem_sample: double    # standard error of the mean
#     val_ll_mean_sample: double
#     val_ll_sem_sample: double
#     test_ll_mean_sample: double
#     test_ll_sem_sample: double

#     tracker_output: attach@external
#     eval_output: attach@external
#     model: attach@external
#     """

#     def make(self, key):
#         print("Received key ->", key)
#         # get posterior args
#         posterior_args = (SBVGPConfig & {"id": key["sbvp_id"]}).fetch1()
#         posterior_args["dist"] = "gamma"

#         # get the FPSamples and MLPCondSamples keys
#         FPSamples_key = (
#             FPSamplesConfig
#             & {
#                 "fp_id": key["fp_samples_id"],
#                 "dl_id": key["dl_id"],
#                 "trainer_id": key["trainer_id"],
#                 "seed": key["data_seed"],
#                 "n_samples": key["n_samples"],
#             }
#         ).fetch1()
#         # Use MLPCondSamplesConfig2 instead of MLPCondSamplesConfig
#         MLPCondSamples_key = (
#             MLPCondSamplesConfig2 & {"ll_id": key["mlpcond_samples_id"]}
#         ).fetch1()
#         # Use MLPCondSamples2 instead of MLPCondSamples
#         prior_samples_path, cond_samples_path = fetch_prior_cond_samples_path(
#             FPSamples, FPSamples_key, MLPCondSamples2, MLPCondSamples_key
#         )

#         # get data_loader args
#         data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
#         data_loader_args["sampled_responses_path"] = prior_samples_path
#         data_loader_args["sampled_obs_path"] = cond_samples_path
#         data_loader_args["data_seed"] = key["data_seed"]

#         # get trainer args
#         trainer_args = (SBVGPTrainerConfig & {"id": key["sbvp_trainer_id"]}).fetch1()

#         if self.FORCE_GPU:
#             if torch.cuda.is_available():
#                 device = torch.device("cuda")
#             else:
#                 raise ValueError("GPU not available.")
#         else:
#             device = torch.device("cpu")
#         trainer_args["device"] = device

#         # train the model
#         (
#             model,
#             train_ll_mean,
#             train_ll_sem,
#             val_ll_mean,
#             val_ll_sem,
#             test_ll_mean,
#             test_ll_sem,
#             train_ll_mean_sample,
#             train_ll_sem_sample,
#             val_ll_mean_sample,
#             val_ll_sem_sample,
#             test_ll_mean_sample,
#             test_ll_sem_sample,
#             tracker_output,
#             eval_output,
#         ) = train_sbvp(data_loader_args, posterior_args, trainer_args, self.USE_WANDB)

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             # save model
#             model_fname = Path(tmp_dir) / f"{make_hash(key)}_model.pt"
#             torch.save(model, model_fname)

#             # save tracker output
#             tracker_output_fname = (
#                 Path(tmp_dir) / f"{make_hash(key)}_tracker_output.pkl"
#             )
#             with open(tracker_output_fname, "wb") as f:
#                 pickle.dump(tracker_output, f)

#             # save eval output
#             eval_output_fname = Path(tmp_dir) / f"{make_hash(key)}_eval_output.pkl"
#             with open(eval_output_fname, "wb") as f:
#                 pickle.dump(eval_output, f)

#             # insert results
#             self.insert1(
#                 {
#                     **key,
#                     "train_ll_mean": train_ll_mean,
#                     "train_ll_sem": train_ll_sem,
#                     "val_ll_mean": val_ll_mean,
#                     "val_ll_sem": val_ll_sem,
#                     "test_ll_mean": test_ll_mean,
#                     "test_ll_sem": test_ll_sem,
#                     "train_ll_mean_sample": train_ll_mean_sample,
#                     "train_ll_sem_sample": train_ll_sem_sample,
#                     "val_ll_mean_sample": val_ll_mean_sample,
#                     "val_ll_sem_sample": val_ll_sem_sample,
#                     "test_ll_mean_sample": test_ll_mean_sample,
#                     "test_ll_sem_sample": test_ll_sem_sample,
#                     "tracker_output": tracker_output_fname,
#                     "eval_output": eval_output_fname,
#                     "model": model_fname,
#                 }
#             )
#             print("Results inserted.")


@schema
class AdaptPriorResult(dj.Computed):
    definition = """
    -> AdaptPriorConfig
    -> AdaptPriorTrainer.proj(trainer_id='id')
    -> AltDataLoaderConfig.proj(dl_id='id') 
    ---
    train_marginal_obs_ll_mean: double    # mean per trial, per sample, in nats
    train_marginal_obs_ll_sem: double    # standard error of the mean
    val_marginal_obs_ll_mean: double
    val_marginal_obs_ll_sem: double
    test_marginal_obs_ll_mean: double
    test_marginal_obs_ll_sem: double

    train_prior_ll_mean: double    # mean per trial, per sample, in nats
    train_prior_ll_sem: double    # standard error of the mean
    val_prior_ll_mean: double
    val_prior_ll_sem: double
    test_prior_ll_mean: double
    test_prior_ll_sem: double
    
    tracker_output: attach@external
    eval_output: attach@external
    model: attach@external  # trained joint model NOT just the prior
    """

    USE_WANDB = False
    FORCE_GPU = False

    def make(self, key):
        print("Fetching FlowPriorResult arguments...")
        prior_model_path = (
            FlowPriorResult
            & {
                "fp_id": key["prior_fp_id"],
                "trainer_id": key["prior_trainer_id"],
                "dl_id": key["orig_dl_id"],
            }
        ).fetch1(download_path="/tmp")["model"]
        likelihood_model_path = (
            LikelihoodResult
            & {
                "ll_id": key["likelihood_id"],
                "trainer_id": key["likelihood_trainer_id"],
                "dl_id": key["orig_dl_id"],
            }
        ).fetch1(download_path="/tmp")["model"]

        fp_args = (FlowPriorConfig & {"id": key["prior_fp_id"]}).fetch1()

        model_args = {
            "seed": key["seed"],
            "prior_model_path": prior_model_path,
            "prior_model_depth": fp_args["flow_depth"],
            "prior_model_initial_nonlin": fp_args["flow_initial_nonlin"],
            "prior_model_final_nonlin": fp_args["flow_final_nonlin"],
            "prior_model_nonlin": fp_args["flow_nonlin"],
            "prior_model_base_dist": fp_args["flow_base_dist"],
            "prior_model_affine_type": fp_args["affine_type"],
            "likelihood_model_path": likelihood_model_path,
        }

        trainer_args = (AdaptPriorTrainer & {"id": key["trainer_id"]}).fetch1()
        trainer_args.pop("id")
        data_loader_args = (AltDataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args.pop("id")

        if self.FORCE_GPU:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                raise ValueError("GPU not available.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer_args["device"] = device

        # train the model
        (
            model,
            train_marginal_obs_ll_mean,
            train_marginal_obs_ll_sem,
            val_marginal_obs_ll_mean,
            val_marginal_obs_ll_sem,
            test_marginal_obs_ll_mean,
            test_marginal_obs_ll_sem,
            train_prior_ll_mean,
            train_prior_ll_sem,
            val_prior_ll_mean,
            val_prior_ll_sem,
            test_prior_ll_mean,
            test_prior_ll_sem,
            tracker_output,
            eval_output,
        ) = adapt_prior(
            data_loader_args, model_args, trainer_args, self.USE_WANDB, dj.conn()
        )

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
                    "train_marginal_obs_ll_mean": train_marginal_obs_ll_mean,
                    "train_marginal_obs_ll_sem": train_marginal_obs_ll_sem,
                    "val_marginal_obs_ll_mean": val_marginal_obs_ll_mean,
                    "val_marginal_obs_ll_sem": val_marginal_obs_ll_sem,
                    "test_marginal_obs_ll_mean": test_marginal_obs_ll_mean,
                    "test_marginal_obs_ll_sem": test_marginal_obs_ll_sem,
                    "train_prior_ll_mean": train_prior_ll_mean,
                    "train_prior_ll_sem": train_prior_ll_sem,
                    "val_prior_ll_mean": val_prior_ll_mean,
                    "val_prior_ll_sem": val_prior_ll_sem,
                    "test_prior_ll_mean": test_prior_ll_mean,
                    "test_prior_ll_sem": test_prior_ll_sem,
                    "tracker_output": tracker_output_fname,
                    "eval_output": eval_output_fname,
                    "model": model_fname,
                }
            )
            print("Results inserted.")


# @schema
# class VPostPriorResult(dj.Computed):
#     """
#     Result table for the Variational Gamma Posterior
#     """

#     USE_WANDB = False
#     FORCE_GPU = False

#     definition = """
#     -> VPConfig.proj(vp_id='id')
#     -> VPTrainerConfig.proj(trainer_id='id')
#     -> AltDataLoaderConfig.proj(dl_id='id')
#     ---
#     train_var_marginal_mean: double    # mean per trial, per sample, in nats
#     train_var_marginal_sem: double    # standard error of the mean
#     val_var_marginal_mean: double
#     val_var_marginal_sem: double
#     test_var_marginal_mean: double
#     test_var_marginal_sem: double

#     train_marginal_obs_ll_mean: double    # mean per trial, per sample, in nats
#     train_marginal_obs_ll_sem: double    # standard error of the mean
#     val_marginal_obs_ll_mean: double
#     val_marginal_obs_ll_sem: double
#     test_marginal_obs_ll_mean: double
#     test_marginal_obs_ll_sem: double

#     train_post_ll_mean: double    # mean per trial, per sample, in nats
#     train_post_ll_sem: double    # standard error of the mean
#     val_post_ll_mean: double
#     val_post_ll_sem: double
#     test_post_ll_mean: double
#     test_post_ll_sem: double

#     train_prior_ll_mean: double    # mean per trial, per sample, in nats
#     train_prior_ll_sem: double    # standard error of the mean
#     val_prior_ll_mean: double
#     val_prior_ll_sem: double
#     test_prior_ll_mean: double
#     test_prior_ll_sem: double

#     tracker_output: attach@external
#     eval_output: attach@external
#     model: attach@external  # trained variational model (with joint and posterior)
#     """

#     def make(self, key):
#         print(
#             f"{self.__class__.__name__}'s {inspect.currentframe().f_code.co_name} called..."
#         )
#         print("Received key ->", key)

#         # get model args
#         vp_args = (VPostPriorConfig & {"id": key["vp_id"]}).fetch1()

#         prior_model_path = (
#             FlowPriorResult
#             & {
#                 "fp_id": key["prior_fp_id"],
#                 "trainer_id": key["prior_trainer_id"],
#                 "dl_id": key["orig_dl_id"],
#             }
#         ).fetch1(download_path="/tmp")["model"]
#         likelihood_model_path = (
#             LikelihoodResult
#             & {
#                 "ll_id": key["likelihood_id"],
#                 "trainer_id": key["likelihood_trainer_id"],
#                 "dl_id": key["orig_dl_id"],
#             }
#         ).fetch1(download_path="/tmp")["model"]

#         fp_args = (FlowPriorConfig & {"id": key["prior_fp_id"]}).fetch1()

#         model_args = {
#             "seed": key["seed"],
#             "prior_model_path": prior_model_path,
#             "prior_model_depth": fp_args["flow_depth"],
#             "prior_model_initial_nonlin": fp_args["flow_initial_nonlin"],
#             "prior_model_final_nonlin": fp_args["flow_final_nonlin"],
#             "prior_model_nonlin": fp_args["flow_nonlin"],
#             "prior_model_base_dist": fp_args["flow_base_dist"],
#             "prior_model_affine_type": fp_args["affine_type"],
#             "likelihood_model_path": likelihood_model_path,
#             "post_dist_type": vp_args["post_dist_type"],
#             "post_nonneg_transform": vp_args["post_nonneg_transform"],
#             "post_n_layers": vp_args["post_n_layers"],
#             "post_nonlin": vp_args["post_nonlin"],
#             "post_dropout_rate": vp_args["post_dropout_rate"],
#             "post_init_std": vp_args["post_init_std"],
#             "post_kwargs": vp_args["post_kwargs"],
#         }

#         trainer_args = (VPTrainerConfig & {"id": key["trainer_id"]}).fetch1()
#         trainer_args.pop("id")

#         # get dataloader args
#         data_loader_args = (AltDataLoaderConfig & {"id": key["dl_id"]}).fetch1()
#         data_loader_args.pop("id")

#         if self.FORCE_GPU:
#             if torch.cuda.is_available():
#                 device = torch.device("cuda")
#             else:
#                 raise ValueError("GPU not available.")
#         else:
#             device = torch.device("cpu")

#         trainer_args["device"] = device

#         # train the model
#         (
#             variational_model,
#             train_var_marginal_mean,
#             train_var_marginal_sem,
#             val_var_marginal_mean,
#             val_var_marginal_sem,
#             test_var_marginal_mean,
#             test_var_marginal_sem,
#             train_marginal_obs_ll_mean,
#             train_marginal_obs_ll_sem,
#             val_marginal_obs_ll_mean,
#             val_marginal_obs_ll_sem,
#             test_marginal_obs_ll_mean,
#             test_marginal_obs_ll_sem,
#             train_post_ll_mean,
#             train_post_ll_sem,
#             val_post_ll_mean,
#             val_post_ll_sem,
#             test_post_ll_mean,
#             test_post_ll_sem,
#             train_prior_ll_mean,
#             train_prior_ll_sem,
#             val_prior_ll_mean,
#             val_prior_ll_sem,
#             test_prior_ll_mean,
#             test_prior_ll_sem,
#             tracker_output,
#             eval_output,
#         ) = train_vpost_prior(
#             data_loader_args, model_args, trainer_args, self.USE_WANDB, dj.conn()
#         )

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             # save model
#             model_fname = Path(tmp_dir) / f"{make_hash(key)}_model.pt"
#             torch.save(variational_model, model_fname)

#             # save tracker output
#             tracker_output_fname = (
#                 Path(tmp_dir) / f"{make_hash(key)}_tracker_output.pkl"
#             )
#             with open(tracker_output_fname, "wb") as f:
#                 pickle.dump(tracker_output, f)

#             # save eval output
#             eval_output_fname = Path(tmp_dir) / f"{make_hash(key)}_eval_output.pkl"
#             with open(eval_output_fname, "wb") as f:
#                 pickle.dump(eval_output, f)

#             # insert results
#             self.insert1(
#                 {
#                     **key,
#                     "train_var_marginal_mean": train_var_marginal_mean,
#                     "train_var_marginal_sem": train_var_marginal_sem,
#                     "val_var_marginal_mean": val_var_marginal_mean,
#                     "val_var_marginal_sem": val_var_marginal_sem,
#                     "test_var_marginal_mean": test_var_marginal_mean,
#                     "test_var_marginal_sem": test_var_marginal_sem,
#                     "train_marginal_obs_ll_mean": train_marginal_obs_ll_mean,
#                     "train_marginal_obs_ll_sem": train_marginal_obs_ll_sem,
#                     "val_marginal_obs_ll_mean": val_marginal_obs_ll_mean,
#                     "val_marginal_obs_ll_sem": val_marginal_obs_ll_sem,
#                     "test_marginal_obs_ll_mean": test_marginal_obs_ll_mean,
#                     "test_marginal_obs_ll_sem": test_marginal_obs_ll_sem,
#                     "train_post_ll_mean": train_post_ll_mean,
#                     "train_post_ll_sem": train_post_ll_sem,
#                     "val_post_ll_mean": val_post_ll_mean,
#                     "val_post_ll_sem": val_post_ll_sem,
#                     "test_post_ll_mean": test_post_ll_mean,
#                     "test_post_ll_sem": test_post_ll_sem,
#                     "train_prior_ll_mean": train_prior_ll_mean,
#                     "train_prior_ll_sem": train_prior_ll_sem,
#                     "val_prior_ll_mean": val_prior_ll_mean,
#                     "val_prior_ll_sem": val_prior_ll_sem,
#                     "test_prior_ll_mean": test_prior_ll_mean,
#                     "test_prior_ll_sem": test_prior_ll_sem,
#                     "tracker_output": tracker_output_fname,
#                     "eval_output": eval_output_fname,
#                     "model": model_fname,
#                 }
#             )
#             print("Results inserted.")
