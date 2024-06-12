def fetch_prior_cond_samples_path(prior_table, prior_key, cond_table, cond_key):
    """
    Fetch the sample paths for prior and conditional tables.

    This function retrieves the sample file paths for the specified prior and
    conditional tables based on provided keys. The sample paths are downloaded
    to a temporary directory and then returned.

    Parameters:
    prior_table (object): The table containing prior samples.
    prior_key (dict): The key used to identify the specific prior sample in the prior_table.
    cond_table (object): The table containing conditional samples.
    cond_key (dict): The key used to identify the specific conditional sample in the cond_table.

    Returns:
    tuple: A tuple containing the paths to the prior and conditional samples.
    """

    # Fetch the prior sample path based on the prior_key
    prior_samples_path = (prior_table & prior_key).fetch1(download_path="/tmp")[
        "samples"
    ]

    # Fetch the conditional sample path based on the cond_key
    cond_samples_path = (cond_table & {**cond_key, **prior_key}).fetch1(
        download_path="/tmp"
    )["samples"]

    return prior_samples_path, cond_samples_path


def fetch_best_model_results(
    result_table,
    config_table,
    data_loader_config_table,
    trainer_config_table,
    config_proj_col,  # Projection column for the config_table
    criterion="val_ll_mean",
    k=1,
    download_path="/tmp",
):
    """
    Fetch the best model results based on a specified criterion.

    This function retrieves the best model results by joining the result table
    with configuration tables, data loader configuration tables, and trainer
    configuration tables based on a specified projection column. The results
    are sorted by a given criterion in descending order, and the top k results
    are returned.

    Parameters:
    result_table (dj table): The result table containing model results.
    config_table (dj table): The configuration table containing model configuration.
    data_loader_config_table (dj table): The data loader configuration table.
    trainer_config_table (dj table): The trainer configuration table.
    config_proj_col (str): The column in the config_table to project on.
    criterion (str, optional): The criterion to sort the results by. Defaults to "val_ll_mean".
    k (int, optional): The number of top results to return. Defaults to 1.
    download_path (str, optional): The path to download the results. Defaults to "/tmp".

    Returns:
    dict or list: A dictionary of the best result if k=1, otherwise a list of dictionaries
    containing the top k results.
    """

    # Get the columns for each table, excluding the 'id' column
    cols = [col for col in config_table.heading if col != "id"]
    dl_cols = [col for col in data_loader_config_table.heading if col != "id"]
    trainer_cols = [col for col in trainer_config_table.heading if col != "id"]

    # Create the combined result table with dynamic projection column
    result_table = (
        result_table
        * config_table.proj(
            *cols, **{config_proj_col: "id"}
        )  # Project configuration table
        * data_loader_config_table.proj(
            dl_id="id", *dl_cols
        )  # Project data loader config table
        * trainer_config_table.proj(
            trainer_id="id", *trainer_cols
        )  # Project trainer config table
    )

    # Fetch the best results ordered by the specified criterion in descending order
    best_val_results = result_table.fetch(
        download_path=download_path, order_by=f"{criterion} DESC", as_dict=True, limit=k
    )

    # Return the results
    # If k == 1, return the single best result as a dictionary
    # Otherwise, return a list of dictionaries containing the top k results
    if k == 1:
        ret = best_val_results[0]
    else:
        ret = best_val_results

    return ret


def drop_schema_dot_jobs(schema):
    schema.jobs.drop()
