"""This script computes summary statistics on the reading behavior of survey 
respondents during the experimental part of the survey. The results are saved 
as .csv files in *bld/analysis/information*.

.. list-table:: Summary statistics over reading behavior
    :header-rows: 1
    :widths: 50 50

    * - File name
      - Summary statistics for ...
    * - ``reading_behavior.csv``
      - Time spent reading.
    * - ``read_attention.csv``
      - Reading behavior conditional on people's attention to the experimental text, proxied by text comprehension question.
    * - ``read_nothing.csv``
      - Characteristics of people who read nothing besides the experimental text.
    * - ``clicks_vs_frictions.csv``
      - Results of Logit models for relationship between survey respondents' information frictions by topic and whether they click on the corresponding text.

"""

import pytask
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.descriptive_stats import (
    compute_stats_by_treatment, melt_friction_vs_clicks, compute_mean_and_ttest
)

depends_on = BLD / "data" / "survey_data.csv"
produces = {
    "reading": BLD / "analysis" / "reading" / "reading_behavior.csv",
    "clicks": BLD / "analysis" / "reading" / "clicks_vs_frictions.csv",
    "read_attention": BLD / "analysis" / "reading" / "read_attention.csv",
    "read_nothing": BLD / "analysis" / "reading" / "read_nothing.csv",
    }

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_summary_reading_behavior(depends_on, produces):

    # survey respondents who provided some outcome
    survey_df = pd.read_csv(depends_on, sep=";").query("any_outcome == 1")

    # some extra variables needed on-the-fly
    survey_df["friction_floodmaps_overestimate"] = np.where(
        survey_df["friction_floodmaps"] > 0, 1, 0)
    survey_df["friction_floodmaps_underestimate"] = np.where(
        survey_df["friction_floodmaps"] < 0, 1, 0)
    survey_df["max_flood_risk"] = np.where(survey_df["correct_floodmaps"] == '1 in 100 years', 1, 0)

    # ex-ante value of experimental text, comparison between those
    # who pass and those who do not pass the attention check
    info_value_dfs = []
    for i, info in enumerate(["nature", "maps", "compensation", "insurance"]):
        treatment = i + 1
        # just get ex-ante value of experimental text
        info_value_df = survey_df.query("treatment == @treatment")[
            [f"info_value_{info}", "uniqueadd_id"]].rename(columns={f"info_value_{info}": "info_value"})
        info_value_dfs.append(info_value_df)
    
    survey_df = survey_df.merge(pd.concat(info_value_dfs), on="uniqueadd_id")

    reading_times = [
        "total_seconds_treatment_text_wins975",
        "total_seconds_all_texts_wins975",
        "clicked_boxes_0",
        "clicked_boxes_3",
        "read_all_flood_texts",
        "text_easy_to_understand_numeric",
        "info_value"
        ]
    read_by_attention = pd.concat(
        [
            compute_mean_and_ttest(survey_df, time, "text_attention_passed")
            for time in reading_times
        ],
        axis=1).T
    
    # attention versus education
    read_by_attention_edu = compute_mean_and_ttest(
        survey_df, "text_attention_passed", "edu_hbo_or_higher")

    # attention versus flood risk category for treatment arm t2
    read_by_attention_t2 = compute_mean_and_ttest(
        survey_df.query("treatment == 2"), "text_attention_passed", "max_flood_risk")
    
    # do the dame but for those who click nothing in treatment 2 vs. those who click something
    # spoiler: also not more likely to overestimate flood risk or to come from safest areas
    t2_df = survey_df.query("treatment == 2")
    dimensions_of_interest = [
        "friction_floodmaps_overestimate",
        "friction_floodmaps_underestimate",
        "flood_max_1_in_10000_years",
        "flood_max_1_in_100_years"
        ]
    read_nothing = pd.concat(
        [
            compute_mean_and_ttest(t2_df, dim, "clicked_boxes_0")
            for dim in dimensions_of_interest
        ],
        axis=1).T

    # relevant columns
    texts = ["decoy", "maps", "wts", "insurance"]

    clicks = [f"clicks_{text}_indicator" for text in texts]
    seconds = [f"conditional_total_seconds_{text}" for text in texts]
    seconds_wins975 = [f"{c}_wins975" for c in seconds]
    seconds_wins99 = [f"{c}_wins99" for c in seconds]
    seconds_columns = [seconds, seconds_wins99, seconds_wins975]

    # clicks by treatment (includes whether attention check is passed)
    clicks_stats = clicks + ["clicked_boxes_0", "read_all_flood_texts", "text_attention_passed"]
    clicks_df = (survey_df.groupby("treatment")[clicks_stats]
        .mean().round(2).T.rename(columns={
            1: "decoy",
            2: "maps",
            3: "WTS",
            4: "insurance"
        }))

    # reading time by treatment
    seconds_df = compute_stats_by_treatment(survey_df, seconds_columns)

    # concatenate
    reading_df = pd.concat([clicks_df, seconds_df])

    # do information frictions on a topic predict clicking on the topic?
    # vanilla spec. (please note: logit does not converge with FE at the participant level)
    t1_df = melt_friction_vs_clicks(survey_df.query("treatment == 1"), ["maps", "wts", "insurance"])
    mod = smf.logit(formula='click ~ friction', data=t1_df)
    res_vanilla = mod.fit()
    res_vanilla_df = res_vanilla.summary2().tables[1]
    res_vanilla_df[["nobs", "llr_pvalue"]] = [res_vanilla.nobs, res_vanilla.llr_pvalue]

    # with clustered errors
    res_clustered = mod.fit(cov_type='cluster', cov_kwds={'groups': t1_df['topic']})
    res_clustered_df = res_clustered.summary2().tables[1]
    res_clustered_df[["nobs", "llr_pvalue"]] = [res_clustered.nobs, res_clustered.llr_pvalue]

    # only for non-default people
    t1_df_nd = melt_friction_vs_clicks(
        survey_df.query("treatment == 1 and clicked_how_many_boxes == 1 or clicked_how_many_boxes == 2"),
        ["maps", "wts", "insurance"]
        )
    mod = smf.logit(formula='click ~ friction', data=t1_df_nd)
    res_nondefault = mod.fit(cov_type='cluster', cov_kwds={'groups': t1_df_nd['topic']})
    res_nondefault_df = res_nondefault.summary2().tables[1]
    res_nondefault_df[["nobs", "llr_pvalue"]] = [res_nondefault.nobs, res_nondefault.llr_pvalue]

    # all together
    click_friction_df = pd.concat(
        [
            res_vanilla_df,
            res_clustered_df,
            res_nondefault_df
        ],
        axis=1,
        keys=["vanilla", "clustered", "non_default_clustered"]
    )

    # save
    reading_df.to_csv(produces["reading"], sep=";")
    click_friction_df.to_csv(produces["clicks"], sep=";")
    read_by_attention.to_csv(produces["read_attention"], sep=";")
    read_nothing.to_csv(produces["read_nothing"], sep=";")