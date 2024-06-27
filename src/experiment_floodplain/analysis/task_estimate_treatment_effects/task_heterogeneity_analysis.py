"""This script estimates heterogeneous treatment effects for a few pre-specified variables. The dataframes of results 
are saved as .csv files in *bld/analysis/heterogeneity*. In each file, each row represents an independent variable,
while each column represents a dependent variable. These files are:

.. list-table:: Heterogeneous treatment effects
   :header-rows: 1
   :widths: 50 50

   * - File name
     - Results for interaction with...
   * - ``interaction_belief_confidence.csv``
     - Confidence in prior beliefs (standardized), for each belief. 
   * - ``interaction_prior_beliefs.csv``
     - Prior beliefs, for each belief. 
   * - ``interaction_confidently_wrong.csv``
     - Indicator for respondents who give at least one confidently wrong answer. 
       plus the two willingness-to-pay measures.
   * - ``interaction_total_frictions.csv``
     - Variable counting the number of incorrect answer in the survey section on information-based questions. 
   * - ``interaction_frictions_by_topic.csv``
     - Indicator for information friction in a certain topic (flood maps, government compensation, insurance).
   * - ``interaction_belief_update.csv``
     - Belief update, for each belief.
   * - ``interaction_belief_update_any.csv``
     - Indicator for whether the respondent update their beliefs, for each belief.
   * - ``interaction_flood_experience.csv``
     - Indicator for self-reported experience of flood damages in 2021.

"""
import pytask
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.regressions import (
    wls_on_beliefs_percentile, wls_with_interaction
)

depends_on = {
    # survey respondents
    "survey_data": BLD / "data" / "survey_data.csv",
    # outcomes and formulas
    "formulas": SRC / "analysis" / "csv" / "formulas.csv"
}
het_path = BLD / "analysis"/ "heterogeneity"
produces = {
    "belief_confidence": het_path / "interaction_belief_confidence.csv",
    "prior_beliefs": het_path/ "interaction_prior_beliefs.csv",
    "confidently_wrong": het_path / "interaction_confidently_wrong.csv",
    "total_frictions": het_path / "interaction_total_frictions.csv",
    "friction_topic": het_path / "interaction_frictions_by_topic.csv",
    "belief_update": het_path / "interaction_belief_update.csv",
    "belief_update_any": het_path / "interaction_belief_update_any.csv",
    "flood_experience": het_path / "interaction_flood_experience.csv",
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_heterogeneity_analysis(depends_on, produces):

    # load data
    survey_df = pd.read_csv(depends_on["survey_data"], sep=";").query("any_outcome == 1")

    # convert malformed WTP to np.nan, so they are dropped from the analysis
    survey_df["wtp_info"] = np.where(survey_df["wtp_info"] == -99, np.nan, survey_df["wtp_info"])
    
    # only one observation with waterdepth higher than 5m
    # adjust this as we use `correct_waterdepth` as control
    survey_df.correct_waterdepth = survey_df.correct_waterdepth.replace(
        {"between 2 and 5m": "more than 2m", "more than 5m": "more than 2m"}
        )
    
    # add absolute size of belief update
    for belief in ["risk", "damages_1000", "comptot", "compshare"]:
        # compute absolute updates
        survey_df[f"{belief}_update_abs"] = np.abs(survey_df[f"{belief}_update"])

    # extract covariates
    formulas_df = pd.read_csv(depends_on["formulas"], sep=";")
    covariates = formulas_df.query(f"VARTYPE == 'PRE_COV'").VARNAME.tolist()

    # arguments (TO BE MOVED TO JSON)
    args_dict = {
        "prior_beliefs": {"outcomes": [
            "risk_standardized", 
            "damages_1000_standardized", 
            "comptot_standardized", 
            "compshare_standardized",
            "worry_numeric", 
            "prior_beliefs_zscore"
            ]},
        "belief_update": {"outcomes": [
            "risk_update",
            "damages_1000_update",
            "comptot_update",
            "compshare_update",
            "worry_numeric_update"
        ]},
        "update_any": {"outcomes": [
            "risk_update_any",
            "damages_update_any",
            "comptot_update_any",
            "compshare_update_any",
            "worry_numeric_update_any"
        ]},
        "confidently_wrong": {"outcomes": [
            "risk_update_abs", 
            "damages_1000_update_abs", 
            "comptot_update_abs", 
            "compshare_update_abs",
            "wtp_insurance_wins99", 
            "wtp_info"]}, 
        "total_frictions": {"outcomes": [
            "risk_update_abs", 
            "damages_1000_update_abs", 
            "comptot_update_abs", 
            "compshare_update_abs",
            "wtp_insurance_wins99", 
            "wtp_info"]}, 
        "flood_experience": {"outcomes": [
            "risk_update", 
            "damages_1000_update", 
            "comptot_update", 
            "compshare_update",
            "worry_numeric_update", 
            "wtp_insurance_wins99", 
            "wtp_info"]},
    }
    
    res_dict = {}

    # interaction effect with prior beliefs
    interactions = args_dict["prior_beliefs"]["outcomes"]

    res_dict["prior_beliefs"] = pd.concat(
        [wls_with_interaction(survey_df, "wtp_insurance_wins99", covariates, interaction)
        for interaction in interactions
    ], keys=interactions, axis=1).round(2)

    # interaction effect with belief updates
    interactions = args_dict["belief_update"]["outcomes"]
    pd.concat(
        [wls_with_interaction(survey_df, "wtp_insurance_wins99", covariates, interaction)
        for interaction in interactions
    ], keys=interactions, axis=1).round(2).to_csv(produces["belief_update"], sep=";")
        
    # interaction effect with belief updates (indicator)
    interactions = args_dict["update_any"]["outcomes"]
    pd.concat(
        [wls_with_interaction(survey_df, "wtp_insurance_wins99", covariates, interaction)
        for interaction in interactions
    ], keys=interactions, axis=1).round(2).to_csv(produces["belief_update_any"], sep=";")

    # confidence in beliefs
    for belief in ["risk", "damages", "comptot", "compshare"]:
        survey_df[f"{belief}_conf_standardized"] = (
            (survey_df[f"{belief}_conf"] - survey_df[f"{belief}_conf"].mean()) / survey_df[f"{belief}_conf"].std())
    outcomes = ["risk_update_abs", "damages_1000_update_abs", "comptot_update_abs", "compshare_update_abs"]
    interactions = ["risk_conf_standardized", "damages_conf_standardized", "comptot_conf_standardized", "compshare_conf_standardized"]
    res_dict["belief_confidence"] = pd.concat(
        [wls_with_interaction(survey_df, outcome, covariates, interaction) 
        for interaction, outcome in zip(interactions, outcomes)], axis=1, keys=outcomes).round(2)

    # confidently wrong on absolute updates and wtp
    outcomes = args_dict["confidently_wrong"]["outcomes"]
    res_dict["confidently_wrong"] = pd.concat(
        [wls_with_interaction(survey_df, outcome, covariates, "confidently_wrong_indicator") 
        for outcome in outcomes], axis=1, keys=outcomes).round(2)

    # total information frictions
    outcomes =  args_dict["total_frictions"]["outcomes"]
    res_dict["total_frictions"] = pd.concat(
        [wls_with_interaction(survey_df, outcome, covariates, "total_frictions") 
        for outcome in outcomes], axis=1, keys=outcomes).round(2)
      
    # frictions by topic
    survey_df["friction_comp"] = np.where((survey_df["WTS_numeric"] == 1) | (survey_df["WTScomp_numeric"] == 1), 1, 0)
    outcomes = ["risk_update", "damages_1000_update", "comptot_update", "compshare_update"]
    interactions = ["friction_floodmaps", "friction_waterdepth", "friction_comp", "friction_comp"]
    res_dict["friction_topic"] = pd.concat(
        [wls_with_interaction(survey_df, outcome, covariates, interaction) 
        for outcome, interaction in zip(outcomes, interactions)], axis=1, keys=outcomes).round(2)
    res_dict["friction_topic"].to_csv(produces["friction_topic"], sep=";")

    # flood experience
    outcomes =  args_dict["flood_experience"]["outcomes"]
    res_dict["flood_experience"] = pd.concat(
        [wls_with_interaction(survey_df, outcome, covariates, "julyflood_damages_yes") 
        for outcome in outcomes], axis=1, keys=outcomes).round(2)
    
    # save everything
    for key in res_dict.keys():
        res_dict[key].to_csv(produces[key], sep=";")