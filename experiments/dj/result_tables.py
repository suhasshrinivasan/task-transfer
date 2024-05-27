import pickle
from pathlib import Path

import datajoint as dj
import torch

from ..learning.train_flow_prior import train_flow_prior
from .dataloader_tables import DataLoaderConfig
from .prior_tables import FlowPriorConfig
from .schema import schema
from .trainer_tables import FPTrainerConfig


@schema
class FlowPriorResult(dj.Computed):
    definition = """
        -> FlowPriorConfig.proj(fp_id='id')
        -> FPTrainerConfig.proj(trainer_id='id')
        -> DataLoaderConfig.proj(dl_id='id')
        ---
        train_ll_mean: float    # mean per dimension, per sample, in bits
        train_ll_sem: float    # standard error of the mean
        val_ll_mean: float
        val_ll_sem: float
        test_ll_mean: float
        test_ll_sem: float
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
