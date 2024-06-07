import pickle
import tempfile
from pathlib import Path

import datajoint as dj
import torch

from ..learning.train_flow_prior import train_flow_prior
from ..learning.train_likelihood import train_likelihood
from ..learning.train_posterior import train_sbvp
from .dataloader_tables import (
    DataLoaderConfig,
    FP_SamplesConfig,
    MLPCond_SamplesConfig,
    fetch_samples_path_from_dj,
)
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
class SBVGPResult(dj.Computed):
    """
    Result table for the Sample Based Variational Gamma Posterior
    """

    definition = """
    -> SBVGPConfig.proj(sbvp_id='id')
    -> SBVGPTrainerConfig.proj(trainer_id='id')
    -> FP_SamplesConfig.proj(fp_samples_id='id')
    -> MLPCond_SamplesConfig.proj(mlpcond_samples_id='id')
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

        # get posterior args
        posterior_args = (SBVGPConfig & {"id": key["sbvp_id"]}).fetch1()
        posterior_args["dist"] = "gamma"
        posterior_args.pop("id")

        # get the FP_Samples and MLPCond_Samples keys
        FP_Samples_key = (FP_SamplesConfig & {"id": key["fp_samples_id"]}).fetch1()
        MLPCond_Samples_key = (
            MLPCond_SamplesConfig & {"id": key["mlpcond_samples_id"]}
        ).fetch1()
        prior_samples_path, cond_samples_path = fetch_samples_path_from_dj(
            FP_Samples_key, MLPCond_Samples_key
        )

        # get data_loader args
        data_loader_args = (DataLoaderConfig & {"id": key["dl_id"]}).fetch1()
        data_loader_args["response_samples_path"] = prior_samples_path
        data_loader_args["obs_samples_path"] = cond_samples_path
        data_loader_args.pop("id")

        # get trainer args
        trainer_args = (SBVGPTrainerConfig & {"id": key["trainer_id"]}).fetch1()
        trainer_args.pop("id")

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
        ) = train_sbvp(data_loader_args, posterior_args, trainer_args)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save model
            model_fname = (
                Path(tmp_dir) / f"{key['sbvp_id']}_{key['trainer_id']}_model.pt"
            )
            torch.save(model, model_fname)

            # save tracker output
            tracker_output_fname = (
                Path(tmp_dir)
                / f"{key['sbvp_id']}_{key['trainer_id']}_tracker_output.pkl"
            )
            with open(tracker_output_fname, "wb") as f:
                pickle.dump(tracker_output, f)

            # save eval output
            eval_output_fname = (
                Path(tmp_dir) / f"{key['sbvp_id']}_{key['trainer_id']}_eval_output.pkl"
            )
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
