import os

import datajoint as dj

dj.config["enable_python_native_blobs"] = True

dj.config["stores"] = {
    "external": {
        "protocol": "s3",
        "endpoint": os.environ["MINIO_ENDPOINT"],
        "access_key": os.environ["MINIO_ACCESS_KEY"],
        "secret_key": os.environ["MINIO_SECRET_KEY"],
        "bucket": "neural-sampling-code",
        "location": "dj-store",
        "secure": True,
    }
}

schema = dj.schema("sshrinivasan_task_transfer")


# @schema
# class JointResults(dj.Computed):

#     prior_config = FlowPriorConfig
#     likelihood_config = LikelihoodConfig
#     trainer_config = JointTrainerConfig
#     data_loader_config = DataLoaderConfig

#     @property
#     def definition(self):
#         return f"""
#             -> {self.prior_config}
#             -> {self.likelihood_config}
#             -> {self.trainer_config}
#             -> {self.data_loader_config}
#             ---
#             trained_model: external
#         """

#     def make(self, key):
#         # Load the data
#         data_cfg = (DataLoaderConfig & key).fetch1()
#         with open(data_cfg["data_fname"], "rb") as f:
#             data = pickle.load(f)
#         train_x = data["x_samples"][
#             : int(data_cfg["n_samples"] * data_cfg["train_prop"])
#         ]
#         train_i = data["i_samples"][
#             : int(data_cfg["n_samples"] * data_cfg["train_prop"])
#         ]
#         val_x = data["x_samples"][
#             int(data_cfg["n_samples"] * data_cfg["train_prop"]) : int(
#                 data_cfg["n_samples"]
#                 * (data_cfg["train_prop"] + data_cfg["val_prop"])
#             )
#         ]
#         val_i = data["i_samples"][
#             int(data_cfg["n_samples"] * data_cfg["train_prop"]) : int(
#                 data_cfg["n_samples"]
#                 * (data_cfg["train_prop"] + data_cfg["val_prop"])
#             )
#         ]
#         test_x = data["x_samples"][
#             int(
#                 data_cfg["n_samples"]
#                 * (data_cfg["train_prop"] + data_cfg["val_prop"])
#             ) :
#         ]
#         test_i = data["i_samples"][
#             int(
#                 data_cfg["n_samples"]
#                 * (data_cfg["train_prop"] + data_cfg["val_prop"])
#             ) :
#         ]
#         train_loader = DataLoader(
#             TensorDataset(train_x, train_i),
#             batch_size=data_cfg["batch_size"],
#             shuffle=True,
#         )
#         val_loader = DataLoader(
#             TensorDataset(val_x, val_i),
#             batch_size=data_cfg["batch_size"],
#             shuffle=False,
#         )
#         test_loader = DataLoader(
#             TensorDataset(test_x, test_i),
#             batch_size=data_cfg["batch_size"],
#             shuffle=False,
#         )

#         # Build the model
#         model_cfg = {
#             "prior": (FlowPriorConfig & key).fetch1(),
#             "likelihood": (LikelihoodConfig & key).fetch1(),
#         }
#         joint = build_joint_model(model_cfg)

#         # Build the trainer
#         training_cfg = (JointTrainerConfig & key).fetch1()
#         trainer =
