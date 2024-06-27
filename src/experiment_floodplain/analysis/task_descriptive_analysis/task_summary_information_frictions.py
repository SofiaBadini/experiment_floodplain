"""This script compute summary statistics for survey respondents' answers to
information-based questions in the survey. The results are saved as .csv files 
in *bld/analysis/information*. These are:

.. list-table:: Summary statistics over information quality
   :header-rows: 1
   :widths: 50 50

   * - File name
     - Summary statistics, questions on...
   * - ``frictions_maps.csv``
     - Flood maps
   * - ``frictions_insurance.csv``
     - Insurance availability.
   * - ``frictions_claims.csv``
     - Claims paid out by the government in July 2021.
   * - ``frictions_wts.csv``
     - WTS act on government compensation.
   * - ``confidently_incorrect_beliefs.csv``
     - Confidence in answers to information-based questions, people who are confidently incorrect
       vs. people who are not.

"""
import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.descriptive_stats import (
    get_maps_frictions, compute_share_by_columns, compute_mean_and_ttest
)

depends_on = BLD / "data" / "survey_data.csv"
info_path = BLD / "analysis" / "information"
produces = {
    "maps": info_path / "frictions_maps.csv",
    "insurance": info_path / "frictions_insurance.csv",
    "claims": info_path / "frictions_claims.csv",
    "wts": info_path / "frictions_wts.csv",
    "confidently_incorrect_beliefs": info_path / "confidently_incorrect_beliefs.csv"
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_summary_information_frictions(depends_on, produces):

    # survey respondents who provided some outcome
    survey_df = pd.read_csv(depends_on, index_col=[0], sep=";").query("any_outcome == 1")

    # dictionaries to be used shortly
    risk_dict = {
        1: "1 in 10,000 years",
        2: "1 in 1,000 years",
        3: "1 in 100 years",
        4: "1 in 10 years"
    }
    water_dict = {
        0: "0 cm",
        1: "< 50cm",
        2: "50cm-1m",
        3: "1m-1.5m",
        4: "1.5m-2m",
        5: "2m-5m",
        6: "> 5m"
    }

    # compute table of correct vs. stated flood maps information
    map_dicts = [risk_dict, water_dict]
    maps = ["floodmaps", "waterdepth"]
    maps_dfs = [get_maps_frictions(survey_df, map, mapdict) for map, mapdict in zip(maps, map_dicts)]
    maps_df = pd.concat(maps_dfs)

    # insurance
    ins_types = ["ins_rain", "ins_secondary", "ins_primary"]
    ins_df = pd.concat([
        pd.DataFrame(survey_df[ins].value_counts()).sort_index()
        for ins in ins_types
        ], axis=1)
    ins_df = compute_share_by_columns(ins_df)
    ins_df.columns = ins_types

    # claims
    claims_df = pd.DataFrame(survey_df["friction_claims"].value_counts())
    claims_df = compute_share_by_columns(claims_df)

    # WTS
    wts_cols = ["WTS", "WTScomp"]
    wts_df = pd.concat([
        compute_share_by_columns(pd.DataFrame(survey_df[wts].value_counts()))
        for wts in wts_cols],
        axis=1)
    wts_df.columns = wts_cols

    # beliefs of confidently incorrect people
    conf_df = compute_mean_and_ttest(survey_df, "average_beliefs_confidence", "confidently_wrong_indicator")

    # save
    maps_df.to_csv(produces["maps"], sep=";")
    ins_df.to_csv(produces["insurance"], sep=";")
    claims_df.to_csv(produces["claims"], sep=";")
    wts_df.to_csv(produces["wts"], sep=";")
    conf_df.to_csv(produces["confidently_incorrect_beliefs"], sep=";")