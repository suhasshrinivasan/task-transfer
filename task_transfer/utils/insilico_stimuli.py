import numpy as np
from insilico_stimuli.stimuli import GaborSet, PlaidsGratingSet


def generate_gabors(
    orientations,
    gabor_params={
        "canvas_size": [12, 12],
        "sizes": [10],
        "spatial_frequencies": [1 / 3],
        "contrasts": [1.0],
        "grey_levels": [0.0],
        "eccentricities": [0.0],
        "locations": [[6, 6]],
        "phases": [np.pi / 2],
        "relative_sf": False,
    },
):
    gabor_params["orientations"] = orientations
    gabor_set = GaborSet(**gabor_params)
    # # visualize the gabor filters
    # n_g = gabor_set.images().shape[0]
    # fig, ax = plt.subplots(1, n_g, dpi=300)
    # for i in range(n_g):
    #     ax[i].imshow(gabor_set.images()[i, :, :], cmap="gray")
    #     ax[i].axis("off")
    return gabor_set.images()


def generate_gratings(
    orientations,
    grating_params={
        "canvas_size": [12, 12],
        "sizes": [10],
        "spatial_frequencies": [1 / 3],
        "contrasts": [1.0],
        "grey_levels": [0.0],
        "eccentricities": [0.0],
        "locations": [[6, 6]],
        "phases": [np.pi / 2],
        "relative_sf": False,
    },
):
    grating_params["orientations"] = orientations
    grating_set = PlaidsGratingSet(**grating_params)
    # # visualize the gabor filters
    # n_g = grating_set.images().shape[0]
    # fig, ax = plt.subplots(1, n_g, dpi=300)
    # for i in range(n_g):
    #     ax[i].imshow(grating_set.images()[i, :, :], cmap="gray")
    #     ax[i].axis("off")
    return grating_set.images()
