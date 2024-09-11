"""This script compute summary statistics for survey respondents' baseline beliefs.
The results are saved as .csv files in *bld/analysis/beliefs*. These are:

.. list-table:: Summary statistics over reported beliefs
   :header-rows: 1
   :widths: 50 50

    
   * - File name
     - Summary statistics of ...
   * - summary_beliefs_all.csv
     - All prior beliefs (10-year flood probability, damages, total and government compensation)
   * - summary_beliefs_quartiles.csv
     - All prior beliefs (mean, median, and standard deviation) by quartile.
   * - summary_beliefs_risk.csv
     - Prior beliefs about 10-year flood probability, by reported and true flood risk categories.
   * - summary_beliefs_damages.csv
     - Prior beliefs about damages, by reported and true maximum water depth categories.
   * - summary_beliefs_t1.csv
     - Prior beliefs, posterior beliefs, and belief updates under treatment 1 of the survey experiment ("neutral text").
   * - summary_beliefs_updates.csv
     - All belief updates.
   * - summary_beliefs_updates_direction.csv
     - Belief updates, by expected direction of the updates (based on baseline information frictions).
   * - summary_risk_updates_direction.csv
     - Belief update over 10-year flood probability, by expected direction of the updates (based on baseline information frictions), 
       tabulated by objective flood risk category.
"""

import pytask
import numpy as np
import pandas as pd
import pathlib

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.descriptive_stats import (
    get_summary_stats, 
    get_conditional_summary_stats, 
    compute_updating_stats, 
    compute_share_of_updates,
    assess_risk_update_direction,
    make_conditional_belief_updates_table
)

depends_on = BLD / "data" / "survey_data.csv"
beliefs_path = BLD / "analysis" / "beliefs"
produces = {
    "all": beliefs_path / "summary_beliefs_all.csv",
    "t1": beliefs_path / "summary_beliefs_t1.csv",
    "risk": beliefs_path / "summary_beliefs_risk.csv",
    "damages": beliefs_path / "summary_beliefs_damages.csv",
    "quartiles": beliefs_path / "summary_beliefs_quartiles.csv",
    "updates":  beliefs_path / "summary_beliefs_updates.csv",
    "direction":  beliefs_path / "summary_beliefs_updates_direction.csv",
    "risk_direction": beliefs_path / "summary_risk_updates_direction.csv"
    }

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_summary_beliefs(depends_on, produces):

    # survey respondents who provided some outcome
    survey_df = pd.read_csv(
        depends_on, index_col=[0], sep=";", encoding="latin1"
        ).query("any_outcome == 1")

    # prior beliefs
    beliefs = [
        "risk",
        "damages_wins975_1000",
        "damages_wins99_1000",
        "damages_1000",
        "comptot",
        "compshare"
        ]

    # posterior beliefs
    beliefs_RE = [
        "risk_RE",
        "damages_RE_wins975_1000",
        "damages_RE_wins99_1000",
        "damages_RE_1000",
        "comptot_RE",
        "compshare_RE"
    ]

    # create "updates" columns
    for belief, belief_RE in zip(beliefs, beliefs_RE):
        survey_df[f"{belief}_update"] = abs(survey_df[belief_RE] - survey_df[belief])

    # create additional column reflecting direction of belief updating
    survey_df = assess_risk_update_direction(survey_df)

    # compute summary statistics of beliefs (unconditional)
    summary_df = get_summary_stats(survey_df, beliefs)

    # compute summary statistics of updating behavior only for T1
    update_cols = [f"{belief}_update" for belief in beliefs]
    t1 = survey_df.query("treatment == 1")
    prior_df = get_summary_stats(t1, beliefs)
    posterior_df = get_summary_stats(t1, beliefs_RE)
    updates_df = get_summary_stats(t1, update_cols)

    t1_df = pd.concat(
        [prior_df, posterior_df, updates_df],
        axis=1,
        keys=["prior", "posterior", "updates"]
    )

    # compute summary statistics of beliefs about risk, conditional on objective flood risk status
    risk_obj = get_conditional_summary_stats(
        df=survey_df,
        col='correct_floodmaps',
        belief="risk",
        vals=['1 in 100 years', '1 in 1000 years', '1 in 10000 years'],
        colnames=['true_risk_100', 'true_risk_1000', 'true_risk_10000']
        )
    risk_subj = get_conditional_summary_stats(
        df=survey_df,
        col='stated_floodmaps_numeric',
        belief="risk",
        vals=[0, 1, 2, 3, 4],
        colnames=['subj_risk_0', 'subj_risk_10000', 'subj_risk_1000', 'subj_risk_100', 'subj_risk_10']
        )
    risk_df = pd.concat([risk_obj, risk_subj], axis=1)

    # compute summary statistics of beliefs about damages,
    # conditional on objective max. water depth
    all_damages = ["damages_wins975_1000", "damages_wins99_1000", "damages_1000"]
    vals = [0, 1, 2, 3, 4, 5]
    waterdepth_categories = ["wt_0cm", "wt_50cm", "wt_1m", "wt_15m", "wt_2m", "wt_5m"]
    obj_waterdepth_categories = [f"obj_{w}" for w in waterdepth_categories]
    subj_waterdepth_categories = [f"subj_{w}" for w in waterdepth_categories]

    # waterdepth categories
    waterdepth_dfs = []
    for colnames in [obj_waterdepth_categories, subj_waterdepth_categories]:
        dfs = [get_conditional_summary_stats(
            df=survey_df,
            col="correct_waterdepth_numeric",
            belief=damages,
            vals=vals,
            colnames=colnames
            ).round(0)
            for damages in all_damages]
        df = pd.concat(dfs, axis=1, keys=all_damages)
        waterdepth_dfs.append(df)

    waterdepth_df = pd.concat(waterdepth_dfs, axis=1)

    # compute mean, median, and SD by quartile
    beliefs = ["risk_RE", "damages_RE_1000", "compshare", "comptot", "wtp_insurance_wins99", "wtp_info"]
    beliefs_data = survey_df.groupby("prior_beliefs_zscore_quartile")[beliefs]
    beliefs_quartiles = pd.concat(
        [beliefs_data.mean(), beliefs_data.median(), beliefs_data.median()],
        axis=1, keys=["mean", "median", "SD"]).T.round(2)

    # compute statistics on updates
    beliefs = ["risk", "damages", "comptot", "compshare", "worry_numeric"]
    how = ["correct", "incorrect"]

    updates_df = pd.concat(
        [compute_updating_stats(survey_df, belief) for belief in beliefs],
        axis=1,
        keys=beliefs)

    beliefs = ["risk", "damages", "compshare"]
    direction_df = pd.concat(
        [
            (pd.DataFrame(survey_df
                .groupby("treatment")[f"{belief}_revise_expected"]
                .value_counts(normalize=True, dropna=False))
                .query(f"{belief}_revise_expected == 1")
                .reset_index()
                .drop(columns=f"{belief}_revise_expected")
                .rename(columns={"proportion": "proportion_expected_updates"})
            ) for belief in beliefs
        ], axis=1, keys=beliefs).round(3)

    # belief updating over 10-year flood probability, by direction and other covariates 
    risk_direction_df = make_conditional_belief_updates_table(
        survey_df, "correct_floodmaps", "risicokaart_flood_probability"
    )

    # save everything
    summary_df.to_csv(produces["all"], sep=";")
    t1_df.to_csv(produces["t1"], sep=";") # only for treatment arm 1
    risk_df.to_csv(produces["risk"], sep=";") # conditional on obj. risk
    waterdepth_df.to_csv(produces["damages"], sep=";") # conditional on obj. water depth
    beliefs_quartiles.to_csv(produces["quartiles"], sep=";") # stats. by quartile
    updates_df.to_csv(produces["updates"], sep=";") # updating stats.
    direction_df.to_csv(produces["direction"], sep=";") # updating stats., share of expected direction
    risk_direction_df.to_csv(produces["risk_direction"], sep=";") # updating stats., 10-year flood prob.