"""This script adds the survey weights to the full sample of addresses eligible
to be contacted. I oversample in areas affected
by the July 2021 floods. The resulting dataset is ``full_sample_with_weights.csv``
in *bld/data*.

.. note::
   The sampling weights will be the true ones only if ``rvo_pc6.csv`` is placed
   under *src/experiment_floodplain/data*. If not, the variable ``FLOODED`` (on which the sampling
   strategy depends) will be assigned at random, thus affecting the sampling weights.
   This does not really matter to reproduce the results in the paper, because
   I provide the dataset of true survey recipients (without identifying information)
   in *bld/replication_data/SURVEY*.

"""
import pytask
import pandas as pd
import numpy as np

from experiment_floodplain.config import BLD

@pytask.mark.depends_on(BLD / "data" / "full_sample.csv")
@pytask.mark.produces(BLD / "data" / "full_sample_with_weights.csv")
def task_add_survey_weights_to_full_sample(depends_on, produces):

    # read full sample
    fullSample = pd.read_csv(depends_on, low_memory=False, index_col=[0], sep=";")

    # get inclusion probabilities, purposefully ovesampling from flooded areas.
    # Each "flooded" observation is 4 times as likely to be sampled as a "not flooded" observation
    floodObsCount = len(fullSample.query("FLOODED == 1"))
    nonfloodObsCount = len(fullSample.query("FLOODED != 1"))
    oversamplingRatio = 4

    # compute inclusion probabilities.
    # non-flooded addresses have inclusion probability of 1.
    inclusionProbabilitiesFlooded = (
        nonfloodObsCount / floodObsCount
    ) / oversamplingRatio
    inclusionProbabilities = np.where(
        fullSample["FLOODED"] == 1, inclusionProbabilitiesFlooded, 1
    )

    # normalize inclusion probabilities (need to sum to 1 across population)
    inclusionProbabilities = inclusionProbabilities / inclusionProbabilities.sum()

    # check if inclusion probabilities sum to 1
    inclusionProbabilities.sum() == 1

    # add inclusion probabilities and sampling weights to `popDF`
    fullSample["INCLUSION_PROBABILITIES"] = inclusionProbabilities
    fullSample["WEIGHTS"] = 1 / inclusionProbabilities

    fullSample.to_csv(produces, sep=";")