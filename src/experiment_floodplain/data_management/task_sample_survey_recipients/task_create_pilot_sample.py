"""This script creates the Qualtrics contact list for the pilot sample. This script 
produces three .csv files, all saved in *bld/data/survey_recipients*:

    - ``pilot_sample.csv``, which contains the sampled addresses.
    - ``pilot_balance.csv``, which compares a number of average covariate values
      between the population and (both unweighted and weighted) sample.
    - ``pilot_qualtrics.csv``, which has the same dimensions of ``pilotSample.csv``
      but only contains variables needed for the qualtrics survey, including unique
      codes to be used as `authenticators <https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/advanced-elements/authenticator/authenticator-overview/>`_.

.. _authenticators: https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/advanced-elements/authenticator/authenticator-overview/

"""
import pytask
import pandas as pd
import geopandas as gpd
import numpy as np

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.data_management.task_sample_survey_recipients.sample import (
    id_generator,
    get_covariates_dataset,
    format_data_for_qualtrics
)

depends_on = {
    "fullSample": BLD / "data" / "full_sample_with_weights.csv",
    "covariates": SRC / "data_management" / "task_sample_survey_recipients" / "csv" / "covariates.csv",
}

pilotPath = BLD / "data" / "survey_recipients" / "pilot"
produces = {
    "pilotSample": pilotPath / "pilot_sample.csv",
    "pilotBalance": pilotPath / "pilot_balance.csv",
    "pilotQualtrics": pilotPath / "pilot_qualtrics.csv",
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_create_pilot_sample(depends_on, produces):

    # load full sample
    fullSample = pd.read_csv(depends_on["fullSample"], sep=";", low_memory=False, index_col=[0])
    
    # sample, with weights.
    # here I did something in retrospect stupid.
    # initially the idea was to get 15_000 respondents and contact,
    # first, 2000 for the pilot, and then the reminaing 13_000 for the
    # main study. But in the end we decided to sample 15_000 people from
    # the full sample, excluding the pilot observations each.

    sample_size = 15_000
    sampleDF = fullSample.sample(
        sample_size, random_state=24, weights=fullSample["INCLUSION_PROBABILITIES"]
    )

    # include authenticator
    ids = [id_generator(seed=i) for i in range(sample_size)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicates in ID list!")

    sampleDF["PASSWORD"] = ids

    # select pilot waves 
    pilot1 = sampleDF.sample(1000, random_state=24)
    mainDF = sampleDF.drop(pilot1.index, axis=0)
    pilot2_low = mainDF.sample(500, random_state=25)
    mainDF = mainDF.drop(pilot2_low.index, axis=0) # drop from main
    pilot2_high = mainDF.sample(500, random_state=26)
    mainDF = mainDF.drop(pilot2_high.index, axis=0) # drop from main

    # check we are not missing any observations
    whole_index = pilot1.index.tolist() + pilot2_low.index.tolist() + pilot2_high.index.tolist() + mainDF.index.tolist()
    len(whole_index) == 15000
    len([i for i in sampleDF.index.tolist() if i not in whole_index]) == 0

    # get pilot dataset
    pilot1["pilot_wave"] = "1"
    pilot2_low["pilot_wave"] = "2_low"
    pilot2_high["pilot_wave"] = "2_high"

    # to save
    pilotDF = pd.concat([pilot1, pilot2_low, pilot2_high])

    # check covariates average values in population vs. pilot data
    covariatesDF = pd.read_csv(depends_on["covariates"], sep=";")
    fullCovDF = get_covariates_dataset(covariatesDF, fullSample)
    pilotCovDF = get_covariates_dataset(covariatesDF, pilotDF)
    pilotReweightedDF = get_covariates_dataset(covariatesDF, pilotDF, weights=True)

    # to save
    balanceDF = pd.concat(
        [fullCovDF, pilotCovDF, pilotReweightedDF],
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
    qualtricsDF = pilotDF[qualtrics_col].copy()
    qualtricsDF = format_data_for_qualtrics(qualtricsDF)

    # save everything
    pilotDF.to_csv(produces["pilotSample"], sep=";")
    balanceDF.to_csv(produces["pilotBalance"], sep=";")
    qualtricsDF.to_csv(produces["pilotQualtrics"], sep=";", index=False)