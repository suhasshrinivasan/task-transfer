import pandas as pd

data = pd.read_json("test_mc_marginal_log_likelihood.json", orient="json")


data


import numpy as np

data["log_errors"] = np.log(data["errors"])
# add jitter to all points in all columns
# data = data + np.random.normal(0, 0.01, data.shape)


import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

plt.figure(dpi=300)
axs = parallel_coordinates(
    data,
    cols=["errors", "prior_dim", "conditional_dim", "mc_sample_size"],
    class_column="errors",
    colormap="seismic",
)
# make errors log scale
# make only the first column log scale
axs.set_yscale("log")
# don't plot legend
axs.get_legend().remove()
# rotate x labels
plt.xticks(rotation=90)


import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

plt.figure(dpi=300)
axs = parallel_coordinates(
    data,
    cols=["errors", "prior_dim", "conditional_dim"],
    class_column="errors",
    colormap="seismic",
)
# make errors log scale
# make only the first column log scale
axs.set_yscale("log")
# don't plot legend
axs.get_legend().remove()
# rotate x labels
plt.xticks(rotation=90)
