"""This script creates the Qualtrics contact list for the main sample. This script produces
three .csv files, all saved in *bld/data/survey_recipients*:

    - ``main_sample.csv``, which contains the sampled addresses.
    - ``main_balance.csv``, which compares a number of average covariate values
      between the population and (both unweighted and weighted) sample.
    - ``main_qualtrics.csv``, which has the same dimensions of ``main_sample.csv``
      but only contains variables needed for the qualtrics survey, again including unique
      codes to be used as `authenticators.

"""

import pytask
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.data_management.task_sample_survey_recipients.sample import (
    id_generator,
    get_covariates_dataset,
    format_data_for_qualtrics
)

depends_on = {
    "fullSample": BLD / "data" / "full_sample_with_weights.csv",
    "pilotSample":  BLD / "data" / "survey_recipients" / "pilot" / "pilot_sample.csv",
    "covariates": SRC / "data_management" / "task_sample_survey_recipients" / "csv" / "covariates.csv"
}
mainPath = BLD / "data" / "survey_recipients" / "main"
produces = {
    "mainSample": mainPath / "main_sample.csv",
    "mainBalance": mainPath / "main_balance.csv",
    "mainQualtrics": mainPath / "main_qualtrics.csv",
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_create_main_sample(depends_on, produces):

    # load pilot and full sample
    pilotSample = pd.read_csv(depends_on["pilotSample"], sep=";", low_memory=False, index_col=[0])
    fullSample = pd.read_csv(depends_on["fullSample"], sep=";", low_memory=False, index_col=[0])

    # remove pilot data from full sample
    effectiveSample = fullSample.drop(pilotSample.index, axis=0)

    # sample, with weights.
    sample_size = 15_000
    sampleDF = effectiveSample.sample(
        sample_size, random_state=241, weights=fullSample["INCLUSION_PROBABILITIES"]
    )

    # include authenticator
    ids = [id_generator(seed=i, restriction=True) for i in range(sample_size)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicates in ID list!")

    sampleDF["PASSWORD"] = ids

    # check covariates average values in population vs. sample data
    covariatesDF = pd.read_csv(depends_on["covariates"], sep=";")
    fullCovDF = get_covariates_dataset(covariatesDF, fullSample)
    sampleCovDF = get_covariates_dataset(covariatesDF, sampleDF)
    sampleReweightedDF = get_covariates_dataset(covariatesDF, sampleDF, weights=True)

    # to save
    balanceDF = pd.concat(
        [fullCovDF, sampleCovDF, sampleReweightedDF],
        axis=1,
        keys=["Full sample", "Survey recip.", "Survey recip. (weighted)"],
    ).swaplevel(axis=1)

    # select variables needed for the qualtrics survey
    scenarios = [10, 100, 1000, 10000]
    floodvars = (
        [f"FLOOD_{s}" for s in scenarios]
        + [f"WATERDEPTH_{s}" for s in scenarios]
        + ["FLOOD_MAX", "WATERDEPTH_MAX"]
    )
    qualtrics_col = ["uniqueadd_id", "PASSWORD"] + floodvars

    # to save
    qualtricsDF = sampleDF[qualtrics_col].copy()
    qualtricsDF = format_data_for_qualtrics(qualtricsDF)

    # save everything
    sampleDF.to_csv(produces["mainSample"], sep=";")
    balanceDF.to_csv(produces["mainBalance"], sep=";")
    qualtricsDF.to_csv(produces["mainQualtrics"], sep=";", index=False)