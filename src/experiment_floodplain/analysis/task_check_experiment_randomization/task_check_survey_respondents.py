"""This script compares survey recipients to survey respondents and the full sample of eligible households, for a range of covariates stored in
``covariates_respondents.csv`` in *analysis/csv*. The results are saved as ``samples_covariates.csv`` in *bld/analysis/covariates*.

"""
import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.descriptive_stats import get_covariates_dataset
from experiment_floodplain.data_management.task_clean_survey_data.clean_data import add_floodmaps_indicators

depends_on = {
    # admin. data of full sample
    "full_sample": BLD / "data" / "full_sample_with_weights.csv",
    # survey recipients
    "survey_recipients": BLD / "replication_data" / "SURVEY" / "survey_recipients.csv",
    # survey respondents
    "survey_data": BLD / "data" / "survey_data.csv",
    # covariates we care about
    "covariates": SRC / "analysis" / "csv" / "covariates_respondents.csv",
}
produces = BLD / "analysis" / "covariates" / "samples_covariates.csv"

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_check_survey_respondents(depends_on, produces):

    # load covariates
    covariates = pd.read_csv(depends_on["covariates"], sep=";")

    # load administrative data of full sample
    full_sample = pd.read_csv(
        depends_on["full_sample"],
        sep=";",
        low_memory=False,
        encoding="latin1"
        )
    
    # get survey recipients
    survey_recipients = pd.read_csv(depends_on["survey_recipients"], sep=";")

    # load data on survey respondents who provided some outcome
    survey_respondents = (pd.read_csv(
        depends_on["survey_data"], index_col=[0], sep=";")
        .query("any_outcome == 1")
    )

    # compute covariates summary stats for survey recipients vs. respondents
    resp = get_covariates_dataset(covariates, survey_respondents)
    resp_weights = get_covariates_dataset(covariates, survey_respondents, weights=True)
    recip = get_covariates_dataset(covariates, survey_recipients)
    recip_weights = get_covariates_dataset(covariates, survey_recipients, weights=True)
    full = get_covariates_dataset(covariates, full_sample)

    # merge dataframes
    covariates_df = pd.concat(
        [resp, resp_weights, recip, recip_weights, full],
        keys=[
            "Survey respond.",
            "Survey respond. (weighted)",
            "Survey recip.",
            "Survey recip. (weighted)",
            "Full sample"
        ],
        axis=1,
    ).swaplevel(axis=1).fillna("")

    # compute number of obs. and add row
    dfs = [
        survey_respondents,
        survey_respondents,
        survey_recipients,
        survey_recipients,
        full_sample
        ]
    nobs = [len(r) for r in (dfs)]
    covariates_df.loc[("N. obs", ""), :] = nobs

    # save dataframe
    covariates_df.to_csv(produces, sep=";")
