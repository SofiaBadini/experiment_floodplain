"""Auxiliary functions to generate sample of survey respondents."""

import numpy as np
import pandas as pd
import string
import random


def get_covariates_dataset(covariatesDF, valuesDF, weights=False):
    """Compute (weighted) average values of variables in `covariatesDF` from
    `valuesDF`. Missing values are automatically excluded.

    Args:
        covariatesDF (pandas.DataFrame): Dataframe of covariates. Need to have
            columns named "CATEGORY", "DESCRIPTION", "VARLEVEL", "VARNAME",
            and "WEIGHTS".
        valuesDF (pandas.DataFrame): Dataframes of covariates' values. Need to
            have a column named "VARNAME" containing the variables in `covariatesDF`.
        weights (Bool): Weights to compute weighted mean. If True, will be
            pulled from `valuesDF.WEIGHTS`. Default is False

    Returns:
        pandas.DataFrame

    """
    meanDict = {}
    desc = "Distance from flooded areas for addresses in non-flooded areas, in meters"

    for i, row in covariatesDF.iterrows():

        if row.DESCRIPTION == desc:
            valuesDF = valuesDF.query("FLOODED == 0")

        key = (f"{row.CATEGORY}", f"{row.DESCRIPTION} ({row.VARLEVEL})")

        if row.VARNAME in valuesDF:

            var = valuesDF[row.VARNAME]
            maskedVar = np.ma.MaskedArray(var, mask=np.isnan(var))
            value = (
                var.mean()
                if weights is False
                else np.ma.average(maskedVar, weights=valuesDF.WEIGHTS)
            )
            meanDict.update({key: value.round(3)})
        else:
            meanDict.update({key: 0.000})

    meanDF = pd.DataFrame(meanDict, index=["Average"]).T

    return meanDF


def id_generator(seed, length=8, restriction=False):
    """Generate random string of given length, given `seed`."""
    random.seed(seed)
    chars = string.ascii_uppercase + string.digits

    if restriction:
        forbidden_chars = "0O1I"
        for char in forbidden_chars: 
            chars=chars.replace(char,"")

    return "".join(random.choice(chars) for _ in range(length))


def format_data_for_qualtrics(qualtricsDF):
    """Format pandas.DataFrame that needs to be uploaded
    as Qualtrics contact list."""

    # English to Dutch dictionary
    waterdepthDict = {
        "less than 0.5m": "minder dan 0,5 m",
        "between 0.5 and 1m": "tussen 0,5 en 1,0 m",
        "between 1 and 1.5m": "tussen 1,0 en 1,5 m",
        "between 1.5 and 2m": "tussen 1,5 en 2,0 m",
        "between 2 and 5m": "tussen 2,0 en 5,0 m",
        "more than 5m": "meer dan 5,0 m",
    }

    # adjust for formatting purposes
    scenarios = [10, 100, 1000, 10000]
    for s in scenarios:
        qualtricsDF[f"FLOOD_{s}"] = np.where(
            qualtricsDF[f"FLOOD_{s}"] == 1, "yes", "no"
        )
        qualtricsDF[f"FLOOD_{s}_NL"] = np.where(
            qualtricsDF[f"FLOOD_{s}"] == "yes", "ja", "nee"
        )
        qualtricsDF[f"WATERDEPTH_{s}"] = np.where(
            qualtricsDF[f"WATERDEPTH_{s}"] == "0m", "", qualtricsDF[f"WATERDEPTH_{s}"]
        )
        qualtricsDF[f"WATERDEPTH_{s}_NL"] = qualtricsDF[f"WATERDEPTH_{s}"].replace(
            waterdepthDict
        )
        qualtricsDF = qualtricsDF.rename(
            columns={
                f"FLOOD_{s}": f"FLOOD_{s}_EN",
                f"WATERDEPTH_{s}": f"WATERDEPTH_{s}_EN",
            }
        )

    # dutch to english
    floodmaxDict = {
        "1 in 100 years": "1 op 100 jaar",
        "1 in 1000 years": "1 op 1000 jaar",
        "1 in 10000 years": "1 op 10000 jaar",
    }
    qualtricsDF = qualtricsDF.rename(columns={"FLOOD_MAX": "FLOOD_MAX_EN"})
    qualtricsDF["FLOOD_MAX_NL"] = (
        qualtricsDF["FLOOD_MAX_EN"].replace(floodmaxDict).copy()
    )

    # WATERDEPTH_MAX to text
    qualtricsDF["WATERDEPTH_MAX"] = qualtricsDF["WATERDEPTH_MAX"].replace(
        {
            1: "less than 0.5m",
            2: "between 0.5 and 1m",
            3: "between 1 and 1.5m",
            4: "between 1.5 and 2m",
            5: "between 2 and 5m",
            6: "more than 5m",
        }
    )
    qualtricsDF["WATERDEPTH_MAX_NL"] = qualtricsDF["WATERDEPTH_MAX"].replace(
        waterdepthDict
    )
    qualtricsDF = qualtricsDF.rename(columns={
        "WATERDEPTH_MAX": "WATERDEPTH_MAX_EN", 
        "PASSWORD": "ExternalDataReference"})
    qualtricsDF["Email"] = "email@email.com"  # mandatory field in qualtrics

    # add extra rows for Qualtrics testing
    ids = [f"password{i}" for i in range(1, 11)]
    passwords = [id.upper() for id in ids]
    extrarows = qualtricsDF.tail(10).copy()
    
    extrarows["uniqueadd_id"] = ids
    extrarows["ExternalDataReference"] = passwords
    
    qualtricsDF = pd.concat([qualtricsDF, extrarows])

    return qualtricsDF