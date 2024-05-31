import pickle
from pathlib import Path

import datajoint as dj
import torch

from task_transfer.evaluation.evaluate_generative_model import evaluate_flow_prior
from task_transfer.ml_lib.data_loading import build_dataloaders

from .dataloader_tables import DataLoaderConfig
from .result_tables import FlowPriorResult
from .schema import schema


@schema
class FlowPriorEval(dj.Computed):
    definition = """
        -> FlowPriorResult
        ---
        corr_sgn_agr_mean_train: double
        corr_sgn_agr_sem_train: double
        corr_sgn_matrix_train: longblob
        corr_mse_mean_train: double
        corr_mse_sem_train: double
        corr_mse_matrix_train: longblob
        corr_mae_mean_train: double
        corr_mae_sem_train: double
        corr_mae_matrix_train: longblob
        corr_pearsonr_train: double
        corr_personr_p_train: double
        corr_spearmanr_train: double
        corr_spearmanr_p_train: double
        corr_kendalltau_train: double
        corr_kendalltau_p_train: double
        fig_corr_hist_train: longblob
        fig_corr_data_train: longblob
        fig_corr_sample_train: longblob
        fig_corr_sgn_agr_train: longblob
        fig_corr_mse_train: longblob
        fig_corr_mae_train: longblob
        fig_corr_hist_train: longblob

        corr_sgn_agr_mean_val: double
        corr_sgn_agr_sem_val: double
        corr_sgn_matrix_val: longblob
        corr_mse_mean_val: double
        corr_mse_sem_val: double
        corr_mse_matrix_val: longblob
        corr_mae_mean_val: double
        corr_mae_sem_val: double
        corr_mae_matrix_val: longblob
        corr_pearsonr_val: double
        corr_personr_p_val: double
        corr_spearmanr_val: double
        corr_spearmanr_p_val: double
        corr_kendalltau_val: double
        corr_kendalltau_p_val: double
        fig_corr_hist_val: longblob
        fig_corr_data_val: longblob
        fig_corr_sample_val: longblob
        fig_corr_sgn_agr_val: longblob
        fig_corr_mse_val: longblob
        fig_corr_mae_val: longblob
        fig_corr_hist_val: longblob

        corr_sgn_agr_mean_test: double
        corr_sgn_agr_sem_test: double
        corr_sgn_matrix_test: longblob
        corr_mse_mean_test: double
        corr_mse_sem_test: double
        corr_mse_matrix_test: longblob
        corr_mae_mean_test: double
        corr_mae_sem_test: double
        corr_mae_matrix_test: longblob
        corr_pearsonr_test: double
        corr_personr_p_test: double
        corr_spearmanr_test: double
        corr_spearmanr_p_test: double
        corr_kendalltau_test: double
        corr_kendalltau_p_test: double
        fig_corr_hist_test: longblob
        fig_corr_data_test: longblob
        fig_corr_sample_test: longblob
        fig_corr_sgn_agr_test: longblob
        fig_corr_mse_test: longblob
        fig_corr_mae_test: longblob
        fig_corr_hist_test: longblob
    """

    def make(self, key):
        print("Evaluating flow prior for key: ", key)
        args = (FlowPriorResult & key).fetch1()
        flow_path = args["model"]
        flow_model = torch.load(flow_path)
        flow_model.eval()

        dl_id = args["dl_id"]
        data_args = (DataLoaderConfig & {"dl_id": dl_id}).fetch1()
        data_path = data_args["data_fname"]
        train_prop = data_args["train_prop"]
        val_prop = data_args["val_prop"]
        batch_size = 128  # doesn't matter for evaluation
        train_loader, val_loader, test_loader = build_dataloaders(
            data_path, train_prop, val_prop, batch_size
        )

        # set constants

        device = "cpu"
        n_samples = 100_000
        density_n_samples = 10_000
        plot_params = dict(
            dims_to_plot=range(45),
            fig_dpi=300,
            linewidth=3,
            tick_length=6,
            tick_width=2,
            fontsize=10,
            plot_xlim=(0, 7),
            plot_ylim=(0, 1),
            density_color="darkblue",
            data_color="darkorange",
            data_alpha=1.0,
            sample_color="darkblue",
            sample_alpha=1.0,
            fig_save_dir=Path("/src/project/figures/learning/"),
        )

        print("Trainset evaluation")
        (
            key["corr_sgn_agr_mean_train"],
            key["corr_sgn_agr_sem_train"],
            key["corr_sgn_matrix_train"],
            key["corr_mse_mean_train"],
            key["corr_mse_sem_train"],
            key["corr_mse_matrix_train"],
            key["corr_mae_mean_train"],
            key["corr_mae_sem_train"],
            key["corr_mae_matrix_train"],
            key["corr_pearsonr_train"],
            key["corr_personr_p_train"],
            key["corr_spearmanr_train"],
            key["corr_spearmanr_p_train"],
            key["corr_kendalltau_train"],
            key["corr_kendalltau_p_train"],
            key["fig_corr_hist_train"],
            key["fig_corr_data_train"],
            key["fig_corr_sample_train"],
            key["fig_corr_sgn_agr_train"],
            key["fig_corr_mse_train"],
            key["fig_corr_mae_train"],
            key["fig_corr_hist_train"],
        ) = evaluate_flow_prior(
            flow_model, train_loader, device, n_samples, density_n_samples, plot_params
        )

        print("Valset evaluation")
        (
            key["corr_sgn_agr_mean_val"],
            key["corr_sgn_agr_sem_val"],
            key["corr_sgn_matrix_val"],
            key["corr_mse_mean_val"],
            key["corr_mse_sem_val"],
            key["corr_mse_matrix_val"],
            key["corr_mae_mean_val"],
            key["corr_mae_sem_val"],
            key["corr_mae_matrix_val"],
            key["corr_pearsonr_val"],
            key["corr_personr_p_val"],
            key["corr_spearmanr_val"],
            key["corr_spearmanr_p_val"],
            key["corr_kendalltau_val"],
            key["corr_kendalltau_p_val"],
            key["fig_corr_hist_val"],
            key["fig_corr_data_val"],
            key["fig_corr_sample_val"],
            key["fig_corr_sgn_agr_val"],
            key["fig_corr_mse_val"],
            key["fig_corr_mae_val"],
            key["fig_corr_hist_val"],
        ) = evaluate_flow_prior(
            flow_model, val_loader, device, n_samples, density_n_samples, plot_params
        )
        print("Testset evaluation")
        (
            key["corr_sgn_agr_mean_test"],
            key["corr_sgn_agr_sem_test"],
            key["corr_sgn_matrix_test"],
            key["corr_mse_mean_test"],
            key["corr_mse_sem_test"],
            key["corr_mse_matrix_test"],
            key["corr_mae_mean_test"],
            key["corr_mae_sem_test"],
            key["corr_mae_matrix_test"],
            key["corr_pearsonr_test"],
            key["corr_personr_p_test"],
            key["corr_spearmanr_test"],
            key["corr_spearmanr_p_test"],
            key["corr_kendalltau_test"],
            key["corr_kendalltau_p_test"],
            key["fig_corr_hist_test"],
            key["fig_corr_data_test"],
            key["fig_corr_sample_test"],
            key["fig_corr_sgn_agr_test"],
            key["fig_corr_mse_test"],
            key["fig_corr_mae_test"],
            key["fig_corr_hist_test"],
        ) = evaluate_flow_prior(
            flow_model, test_loader, device, n_samples, density_n_samples, plot_params
        )

        # convert all figs to pickle
        key["fig_corr_hist_train"] = pickle.dumps(key["fig_corr_hist_train"])
        key["fig_corr_data_train"] = pickle.dumps(key["fig_corr_data_train"])
        key["fig_corr_sample_train"] = pickle.dumps(key["fig_corr_sample_train"])
        key["fig_corr_sgn_agr_train"] = pickle.dumps(key["fig_corr_sgn_agr_train"])
        key["fig_corr_mse_train"] = pickle.dumps(key["fig_corr_mse_train"])
        key["fig_corr_mae_train"] = pickle.dumps(key["fig_corr_mae_train"])
        key["fig_corr_hist_train"] = pickle.dumps(key["fig_corr_hist_train"])
        key["fig_corr_hist_val"] = pickle.dumps(key["fig_corr_hist_val"])
        key["fig_corr_data_val"] = pickle.dumps(key["fig_corr_data_val"])
        key["fig_corr_sample_val"] = pickle.dumps(key["fig_corr_sample_val"])
        key["fig_corr_sgn_agr_val"] = pickle.dumps(key["fig_corr_sgn_agr_val"])
        key["fig_corr_mse_val"] = pickle.dumps(key["fig_corr_mse_val"])
        key["fig_corr_mae_val"] = pickle.dumps(key["fig_corr_mae_val"])
        key["fig_corr_hist_val"] = pickle.dumps(key["fig_corr_hist_val"])
        key["fig_corr_hist_test"] = pickle.dumps(key["fig_corr_hist_test"])
        key["fig_corr_data_test"] = pickle.dumps(key["fig_corr_data_test"])
        key["fig_corr_sample_test"] = pickle.dumps(key["fig_corr_sample_test"])
        key["fig_corr_sgn_agr_test"] = pickle.dumps(key["fig_corr_sgn_agr_test"])
        key["fig_corr_mse_test"] = pickle.dumps(key["fig_corr_mse_test"])
        key["fig_corr_mae_test"] = pickle.dumps(key["fig_corr_mae_test"])
        key["fig_corr_hist_test"] = pickle.dumps(key["fig_corr_hist_test"])

        print("Inserting row")
        self.insert1(key)
        print("Key inserted")
        # free up memory
        Path(flow_path).unlink()
        print("Evaluation complete")
