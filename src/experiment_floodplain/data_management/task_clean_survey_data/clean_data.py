"""This script contains functions used in ``task_clean_survey_data.py`` to clean the original
Qualtrics data in *bld/data/SURVEY*.

"""

import numpy as np
import geopandas as gpd
import pandas as pd
import itertools

import datetime
from datetime import date, timedelta


def clean_variables_block(qdf, rvar_df, fvar_df, block, fun):
    """Clean block of variables in `qdf`.

    Args:
        qdf (Pandas.DataFrame): Dataset of raw Qualtrics variables.
        rvar_df (Pandas.DataFrame): Dataset containing the names of
            the raw Qualtrics variables to clean.
        fvar_df (Pandas.DataFrame): Dataset containing the names of
            the final clean variables.
        block (Pandas.DataFrame): Block of variables to clean
            (BLOCK in `rvar_df` and in `fvar_df`).
        fun (fun): Function to clean the specified block of variable.

    Returns:
        Pandas.DataFrame

    """

    # select subset of variables to clean
    vars = rvar_df.query("BLOCK == @block").VARNAME.tolist()
    vars.append("uniqueadd_id")
    vars_df = qdf[vars].copy()

    # clean variables
    kwargs = {"error": 10} if block == 'INFORMATION' else {}
    vars_df = fun(df=vars_df, **kwargs)

    # check that exactly the expected variables are created
    expected_variables = fvar_df.query("BLOCK == @block").VARIABLE.tolist()
    expected_variables.append("uniqueadd_id")
    created_variables = vars_df.columns.tolist()

    # create message
    unexpected_variables = [var for var in created_variables if var not in expected_variables]
    missing_variables = [var for var in expected_variables if var not in created_variables]
    message = f"Unexpected variables: {unexpected_variables}, missing variables: {missing_variables}"

    assert set(created_variables) == set(expected_variables), message

    # append to list of dataframes
    print(f"{block, fun}: successfully completed")

    return vars_df


def clean_beliefs_data(df):
    """Clean dataframe `df` containing questions of Qualtrics survey related to
    prior and posterion beliefs. The labels for questions related to confidence
    in beliefs are in the dictionary `confDict`.

    """

    # renaming (removing "_1"), the default qualtrics suffix for slider questions
    cols_to_rename = df.columns[df.columns.str.endswith("_1")].tolist()
    new_names = [col[:-2] for col in cols_to_rename]
    df = df.rename(columns=dict(zip(cols_to_rename, new_names)))

    # replacing: worry to worry indicator
    df[["worry_numeric", "worry_RE_numeric"]] = df[["worry", "worry_RE"]].replace(
        {
            "Not worried at all": 1,
            "Not so worried": 2,
            "Somewhat worried": 3,
            "Very worried": 4,
            "Extremely worried": 5,
        }
    )

    # confidence level to numeric (qualtrics save it to string by default)
    conf_cols = df.columns[df.columns.str.contains("_conf")].tolist()
    df[conf_cols] = df[conf_cols].astype("float")

    # read damages as float
    df[["damages", "damages_RE"]] = df[["damages", "damages_RE"]].astype("float")

    # create winsorized columns for damages (prior and posterior)
    df = winsorize(df, ["damages", "damages_RE"], 0.975)
    df = winsorize(df, ["damages", "damages_RE"], 0.99)

    # replace two extreme outliers in damages with max. value exluding those (1M EUR)
    df["damages"] =  np.where(df["damages"] > 1000001, 1000000, df["damages"])
    df["damages_RE"] =  np.where(df["damages_RE"] > 1000001, 1000000, df["damages_RE"])

    # add damages in thousands of euros
    damages = df.columns[df.columns.str.startswith("damages")]
    to_be_removed = ["damages_conf", "damages_conf_RE"] # remove confidence columns
    damages = [d for d in damages if d not in to_be_removed]
    df[damages] = df[damages].astype("float")
    damages_1000 = [f"{d}_1000" for d in damages]
    df[damages_1000] = df[damages] / 1000

    # compute (1) belief updates, (2) standardized beliefs,
    # (3) measures of extremeness in beliefs
    prior_beliefs = [
        "risk",
        "damages",
        "damages_wins99",
        "damages_wins975",
        "damages_1000",
        "damages_wins99_1000",
        "damages_wins975_1000",
        "comptot",
        "compshare",
        ]
    posterior_beliefs = [
        "risk_RE",
        "damages_RE",
        "damages_RE_wins99",
        "damages_RE_wins975",
        "damages_RE_1000",
        "damages_RE_wins99_1000",
        "damages_RE_wins975_1000",
        "comptot_RE",
        "compshare_RE",
        ]

    for prior_belief, posterior_belief in zip(prior_beliefs, posterior_beliefs):
        df[prior_belief] = df[prior_belief].astype("float")
        df[posterior_belief] = df[posterior_belief].astype("float")
        df = compute_belief_update(df, prior_belief, posterior_belief)
        df = standardize_variable(df, prior_belief)
        df = assign_variable_tercile(df, prior_belief)
        df = assign_variable_quartile(df, prior_belief)

    # add update in worry variable (numeric)
    df["worry_numeric_update"] = df["worry_RE_numeric"] - df["worry_numeric"]

    # compute overall prior beliefs z-score from standardized variables
    df = compute_prior_beliefs_zscore(df)
    df = assign_variable_quartile(df, "prior_beliefs_zscore")

    # get average confidence across all information-based questions
    df["average_beliefs_confidence"] = (df[
        ["risk_conf", "damages_conf", "comptot_conf", "compshare_conf"]
        ].mean(axis=1).round(0))

    for belief in ["risk", "damages", "comptot", "compshare", "worry_numeric"]:

        # add whether there is any update at all (preserve nans)
        df[f"{belief}_update_any"] = np.where(
            (df[f"{belief}_update"] != 0) & (df[f"{belief}_update"].notna()),
            1,
            np.where(
                df[f"{belief}_update"].isna(),
                np.nan,
                0
            ))
        # add direction of the update (preserve nans)
        df[f"{belief}_revise"] = np.where(
            df[f"{belief}_update"] > 0, "upward", np.where(
                df[f"{belief}_update"] < 0, "downward", np.where(
                    df[f"{belief}_update"] == 0, "none", "nan"
                )
            ))
        df[f"{belief}_revise"] = df[f"{belief}_revise"].replace("nan", np.nan)

    return df


def standardize_variable(df, var):
        """Standardized variable `var` in Pandas.DataFrame `df`."""
        df[f"{var}_standardized"] = (df[var] - df[var].mean()) / df[var].std()
        return df


def assign_variable_quartile(df, var):
    """Compute and assign quartile to each observation of
    variable `var` in Pandas.DataFrame `df`.
    """
    cond_list = [
        df[var] <= df[var].quantile(0.25),
        (df[var] > df[var].quantile(0.25)) & (df[var] <= df[var].quantile(0.5)),
        (df[var] > df[var].quantile(0.5)) & (df[var] <= df[var].quantile(0.75)),
        df[var] >= df[var].quantile(0.75)
    ]
    choice_list = ["q1", "q2", "q3", "q4"]
    df[f"{var}_quartile"] = np.select(cond_list, choice_list, "nan")
    df[f"{var}_quartile"] = df[f"{var}_quartile"].replace("nan", np.nan)

    return df

def assign_variable_tercile(df, var):
    """Compute and assign tercile to each observation of
    variable `var` in Pandas.DataFrame `df`.
    """
    cond_list = [
        df[var] <= df[var].quantile(0.33),
        (df[var] > df[var].quantile(0.33)) & (df[var] <= df[var].quantile(0.66)),
        df[var] >= df[var].quantile(0.66)
    ]
    choice_list = ["t1", "t2", "t3"]
    df[f"{var}_tercile"] = np.select(cond_list, choice_list, "nan")
    df[f"{var}_tercile"] = df[f"{var}_tercile"].replace("nan", np.nan)

    return df


def compute_belief_update(df, prior_belief, posterior_belief):
    """Compute difference between variables `posterior_belief` and
    `prior_belief` in Pandas.DataFrame `df`.
    """
    df[f"{prior_belief}_update"] = df[posterior_belief] - df[prior_belief]
    return df


def compute_prior_beliefs_zscore(df):
    """Compute z-score of prior beliefs in Pandas.DataFrame `df`.

    The variables I use are:
        - `risk_standardized`: Standardized measure of beliefs about
            10-year flood risk.
        - `damages_wins975_1000_standardized`: Standardized measure
            of beliefs about damages, winsorized at the 97.5% percentile,
            in thoursand of euros.
        - `comptot_standardized`: Standardized measure of beliefs about
            total compensation.
        - `compshare_standardized`: Standardized measure of beliefs
            about compensation from the government.

    The standardized beliefs (i.e. the z-scores) increase with expected
    net damages, i.e. the higher `risk_standardized` and
    `damages_wins975_1000_standardized`, and the lowest `comptot_standardized`
    and `compshare_standardized`, the higher the average z-score.

    """

    df["prior_beliefs_zscore"] = (
        df.risk_standardized
        + df.damages_1000_standardized
        - df.comptot_standardized
        - df.compshare_standardized) / 4

    return df


def clean_info_data(df, error):
    """Clean dataframe `df` containing questions of Qualtrics survey related to
    information quality. The labels for questions related to confidence in
    the answers are in the dictionary `confDict`.

    Args:

        df (pandas.DataFrame): Dataframe of survey responses.
        error (float or int): Percentage points threshold for which a
            participants' estimate of the compensated claims is considered
            incorrect.

    Returns:
        pandas.DataFrame.

    """
    # rename value of information
    df = df.rename(columns={
        "info_value_1": "info_value_maps",
        "info_value_2": "info_value_compensation",
        "info_value_3": "info_value_insurance",
        "info_value_4": "info_value_nature"
    })

    # renaming (removing "_1", the default qualtrics suffix for slider questions)
    cols_to_rename = df.columns[df.columns.str.endswith("_1")].tolist()
    new_names = [col[:-2] for col in cols_to_rename]
    df = df.rename(columns=dict(zip(cols_to_rename, new_names)))

    # confidence level to numeric (qualtrics save it to string by default)
    confCols = df.columns[df.columns.str.contains("_conf")].tolist()
    df[confCols] = df[confCols].astype("float")

    # replacing: insurability to insurability indicator
    insurCols = ["ins_rain", "ins_primary", "ins_secondary"]
    insurColsNumeric = [f"{i}_numeric" for i in insurCols]
    df[insurColsNumeric] = df[insurCols].replace(
        {
            "Never, or almost never": 0,
            "By some insurers": 1,
            "Always, or almost always": 2,
        }
    )

    # replacing: yes/no to 1/0
    df[["WTS_numeric", "WTScomp_numeric"]] = df[["WTS", "WTScomp"]].replace({"Yes": 1, "No": 0})

    # rename stated and correct flood maps categories
    df = df.rename(columns={
        "waterdepth": "stated_waterdepth",
        "floodmaps": "stated_floodmaps",
        "WATERDEPTH_MAX_EN": "correct_waterdepth",
        "FLOOD_MAX_EN": "correct_floodmaps"})

    # create numeric for stated and correct flood maps categories
    df["correct_waterdepth_numeric"] = df["correct_waterdepth"].replace(
        {
            "0 cm": 0,
            "less than 0.5m": 1,
            "between 0.5 and 1m": 2,
            "between 1 and 1.5m": 3,
            "between 1.5 and 2m": 4,
            "between 2 and 5m": 5,
            "more than 5m": 6,
        }
    )
    df["correct_floodmaps_numeric"] = (
        df["correct_floodmaps"].str[5:-6].replace({"100": 3, "1000": 2, "10000": 1})
    )
    df["stated_waterdepth_numeric"] = df["stated_waterdepth"].replace(
        {
            "0 cm": 0,
            "Up to 50 cm": 1,
            "Between 50 cm and 1 m": 2,
            "Between 1 and 1.5 m": 3,
            "Between 1.5 and 2 m": 4,
            "Between 2 and 5 m": 5,
            "More than 5 m": 6,
        }
    )
    df["stated_waterdepth_numeric"] = df["stated_waterdepth_numeric"].astype("float")
    df["stated_floodmaps_numeric"] = df["stated_floodmaps"].replace(
        {
            "Large probability (1 flood every 10 years)": 4,
            "Medium probability (1 flood every 100 years)": 3,
            "Small probability (1 flood every 1000 years)": 2,
            "Extremely small probability (1 flood every 10,000 years)": 1,
            "No flood risk": 0,
        }
    )

    df = create_info_friction_indicators(df, error)

    # add "confidently incorrect" indicators
    questions = [
        "floodmaps",
        "waterdepth",
        "WTS",
        "WTScomp",
        "claims",
        "ins_primary",
        "ins_secondary",
        "ins_rain"
        ]

    conditions = [
        (df[f"friction_{q}_indicator"] == 1) & (df[f"{q}_conf"] >= 8) for q in questions
        ]
    for question, condition in zip(questions, conditions):
        df[f"confidently_wrong_{question}"] = np.where(condition, 1, 0)

    df["confidently_wrong_answers"] = df[
        [f"confidently_wrong_{question}" for question in questions]
        ].sum(axis=1)
    df["confidently_wrong_indicator"] = np.where(
        df["confidently_wrong_answers"] > 0, 1, 0
        )

    return df


def create_info_friction_indicators(df, error):
    """Create variables that take value 0 if survey respondents answered
    a question correctly and 1 otherwise. This serves to aggregated information
    frictions.

    Args:

        df (pandas.DataFrame): Dataframe of survey responses.
        error (float or int): Percentage points threshold for which a
            participants' estimate of the compensated claims is considered
            incorrect.

    Returns:
        pandas.DataFrame.

    """
    # this will take a value from -6 to 6
    # should preserve nans if respondent did not provide a stated water depth
    df["friction_waterdepth"] = df["stated_waterdepth_numeric"] - df["correct_waterdepth_numeric"]

    # this will take a value from -3 to 3
    # should preserve nans if respondent did not provide a stated water depth
    df["friction_floodmaps"] = df["stated_floodmaps_numeric"] - df["correct_floodmaps_numeric"]

    # frictions in insurance information
    df["friction_ins_rain"] = np.where(
        # where "ins_rain" is 2, i.e. the answer was "Always, or almost always",
        # which is correct, the respondent gets no information frictions (0).
        # otherwise, unless the respondent did not answer, the information friction
        # is present (1).
        df["ins_rain_numeric"] == 2,
        0,
        np.where(df["ins_rain_numeric"].isna(), np.nan, 1),
    )

    # as above, but the correct answer is 0 ("Never, or almost never"). So 0 stays 0,
    # 1 stays 1 (as it is wrong), 2 becomes 1 (as the indicator is binary), nans stay nans.
    df["friction_ins_primary"] = np.where(df["ins_primary_numeric"] == 2, 1, df["ins_primary_numeric"])

    # as above, but the correct asnwer is 1 ("By some insurers"). So 1 becomes 0 (no friction),
    # nans stay nans, all else becomes 1 (friction)
    df["friction_ins_secondary"] = np.where(
        df["ins_secondary_numeric"] == 1, 0, np.where(df["ins_rain_numeric"].isna(), np.nan, 1)
    )

    # frictions in goverment compensation (distance from true measure)
    df["friction_claims_difference"] = df["claims"].astype("float") - 53

    # duplicating WTS columns: by default formatted such that correct answer is 0
    # and incorrect is 1
    df["friction_WTS_indicator"] = df["WTS_numeric"].values.copy()
    df["friction_WTScomp_indicator"] = df["WTScomp_numeric"].values.copy()

    # dummy variables: get all friction columns (excluding claims and WTS columns)
    frictions_cols = [
        "friction_floodmaps",
        "friction_waterdepth",
        "friction_ins_rain",
        "friction_ins_primary",
        "friction_ins_secondary"
        ]

    # for all columns besides "friction_claims_difference": 0 remains 0 (no friction), nans remain nans,
    # all else remains 1 (some friction)
    for col in frictions_cols:
        df[f"{col}_indicator"] = np.where(
            (df[col] != 0) & (df[col].notna()), 1, df[col]
        )

    # for claims: assign 1 if distance from true claim is larger than error
    df["friction_claims_indicator"] = np.where(
        (df["friction_claims_difference"] <= error) & (df["friction_claims_difference"] >= -error), 0, 1
    )
    # preserve nans
    df["friction_claims_indicator"] = df["friction_claims_indicator"].where(
        df["friction_claims_difference"].notna(), np.nan
    )

    # add non-numeric for claims estimates
    df["friction_claims"] = np.where(
        df["friction_claims_difference"] > error,
        "overestimate",
        np.where(
            df["friction_claims_difference"] < -error,
            "underestimate",
            "about right"
            )
    )

    # total frictions indicator (sum all the 1s)
    indicator_cols = df.columns[df.columns.str.endswith("indicator")].tolist()
    df["total_frictions"] = df[indicator_cols].sum(axis=1, min_count=8)

    # topic-specific frictions indicator
    df["friction_topic_wts"] = (np.where(
        df[[
            "friction_WTS_indicator",
            "friction_WTScomp_indicator",
            "friction_claims_indicator"
            ]].any(axis=1), 1, 0)
    )

    df["friction_topic_maps"] = (np.where(
        df[[
            "friction_floodmaps_indicator",
            "friction_waterdepth_indicator"
        ]].any(axis=1), 1, 0)
    )

    df["friction_topic_insurance"] = (np.where(
        df[[
            "friction_ins_rain_indicator",
            "friction_ins_secondary_indicator",
            "friction_ins_primary_indicator"
        ]].any(axis=1), 1, 0)
    )

    # get average confidence across all information-based questions
    info = [
        "floodmaps",
        "waterdepth",
        "WTS",
        "WTScomp",
        "claims",
        "ins_rain",
        "ins_primary",
        "ins_secondary",
    ]
    infoconf_cols = [f"{i}_conf" for i in info]
    df["average_info_confidence"] = df[infoconf_cols].mean(axis=1).round(0)

    return df


def clean_tech_data(df):
    """Clean dataframe `df` containing questions of Qualtrics survey related to
    technical fields (e.g.: time of survey completion, answers to consent forms...)

    """
    # some renaiming
    df = df.rename(columns={"Duration (in seconds)": "duration_seconds"})

    # convert consent columns to indicator (1 if consent, 0 if not consent)
    consentCols = ["consent", "consent_followup"]
    consentColsNumeric = [f"{c}_numeric" for c in consentCols]
    df[consentColsNumeric] = df[consentCols].replace(
        {
            "I consent to participate in this survey": 1,
            "I do NOT consent to participate in this survey": 0,
            "I do consent to be contacted via email for a follow-up survey": 1,
            "I do NOT consent to be contacted via email for a follow-up survey": 0,
        }
    )

    return df


def clean_treatment_data(df):
    """Clean dataframe `df` containing questions of Qualtrics survey related to
    experimental treatment(s).

    """
    # rename treatment
    df = df.rename(columns={"treatment_assignment": "treatment"})
    df.treatment = df.treatment.astype("float")

    # replace white trails in column names with underscores
    colsNames = df.columns[df.columns.str.contains(" ")]
    newColsNames = colsNames.str.replace(" ", "_").str.lower()
    df = df.rename(columns=dict(zip(colsNames, newColsNames)))

    # create list of column names referring to (1) time spent on treatment
    # texts and (2) time spent on blog post
    suffixes1 = ["text", "info"]
    suffixes2 = ["first_click", "last_click", "click_count", "page_submit"]
    colnames = [
        f"{suffix1}_time_{suffix2}"
        for suffix1, suffix2 in itertools.product(suffixes1, suffixes2)
    ]
    # add column names related to treatment texts evaluation
    colnames = colnames + [
        "text_eval_1",
        "text_eval_2",
        "text_eval_3",
        "text_attention",
    ]

    for col in colnames:
        # the column names in the list above all refer to the
        # same, individual-specific variables: Survey respondents
        # are all asked these same questions regardless of the
        # treatment arm they are assigned to. However, since
        # individuals are assigned to different treatment arms,
        # Qualtrics store these variables under different names.
        df = merge_same_question_different_treatment(df, col)

    # rename text evaluation column
    df = df.rename(columns=
        {
            "text_eval_1": "text_easy_to_understand",
            "text_eval_2": "text_nice_read",
            "text_eval_3": "text_emotions"
        }
    )

    # create numeric columns for text evaluation
    evalCols = ["text_easy_to_understand", "text_nice_read", "text_emotions"]
    evalColsNumeric = [f"{e}_numeric" for e in evalCols]
    df[evalColsNumeric] = df[evalCols].replace(
        {
            "Not at all": 1,
            "Not so much": 2,
            "Somewhat": 3,
            "Very": 4,
            "Extremely": 5
        }
    )

    # convert timestamp strings to datetime objects
    df = format_timestamps(df)

    # time spent on pages (in seconds) to float
    timecols = df.columns[df.columns.str.contains("_time_")]
    df[timecols] = df[timecols].astype("float")

    # compute total time spent on each toggle box
    tscols = df.columns[df.columns.str.startswith("timestamps")]
    seconds_cols = [s.replace("timestamps", "total_seconds") for s in tscols]
    df[seconds_cols] = df[tscols].apply(compute_time_spent_on_info, axis=1).copy()

    # make adjustment for pilot data
    df = make_adjustment_for_pilot_data(df)

    # add time spent on treatment text to "total_seconds" columns
    df = add_time_spent_on_treatment_text(df)

    # drop total seconds spnt on submit button (conditional on all values being 0)
    if (df["total_seconds_submit"] == 0).all():
        df = df.drop(columns=["total_seconds_submit"])
        seconds_cols = [s for s in seconds_cols if s != "total_seconds_submit"]

    # get time spent on each text conditional on clicking and create two winsorized columns
    # (simply replace 0 columns with np.nan, this helps later on with the analysis)
    conditional_seconds = [f"conditional_{s}" for s in seconds_cols]
    df[conditional_seconds] = np.where(df[seconds_cols] == 0, np.nan, df[seconds_cols])
    df = winsorize(df, conditional_seconds, 0.975)
    df = winsorize(df, conditional_seconds, 0.99)

    # get time spent on treatment text, time spent on other texts, total time spent on texts
    conditions = [
        df["treatment"] == 1,
        df["treatment"] == 2,
        df["treatment"] == 3,
        df["treatment"] == 4
        ]
    choices_ttext = [
        df["total_seconds_decoy"],
        df["total_seconds_maps"],
        df["total_seconds_wts"],
        df["total_seconds_insurance"]
        ]
    choices_other_texts = [
        df["total_seconds_decoy"] +
        df["total_seconds_maps"] +
        df["total_seconds_wts"] +
        df["total_seconds_insurance"] -
        c for c in choices_ttext
    ]

    df["total_seconds_treatment_text"] = np.select(conditions, choices_ttext, np.nan)
    df["total_seconds_other_texts"] = np.select(conditions, choices_other_texts, np.nan)
    df["total_seconds_all_texts"] = df["total_seconds_treatment_text"] + df["total_seconds_other_texts"]

    # winsorize
    cols_to_winsorize = [
        "total_seconds_treatment_text",
        "total_seconds_other_texts",
        "total_seconds_all_texts"
        ]
    df = winsorize(df, cols_to_winsorize, 0.975)
    df = winsorize(df, cols_to_winsorize, 0.99)

    # convert clicks on toggle boxes to integer
    clicks_cols = df.columns[df.columns.str.contains("clicks_")]
    df[clicks_cols] = df[clicks_cols].astype("float")

    # add indicator variable for whether survey respondents clicked on each toggle box
    clicks_cols_indicator = [c + "_indicator" for c in clicks_cols]
    df[clicks_cols_indicator] = np.where(df[clicks_cols] >= 1, 1, df[clicks_cols])

    # add variable for toggle boxes clicked (0, 1, 2, or 3) and make dummies
    df["clicked_how_many_boxes"] = df[clicks_cols_indicator].sum(axis=1)
    clickedDummies = pd.get_dummies(df["clicked_how_many_boxes"], prefix="clicked_boxes", dtype=float)
    clickedDummies.columns = [d[:-2] for d in clickedDummies.columns]
    df = pd.concat([df, clickedDummies], axis=1)

    # in the analysis I assume text assigned is "clicked", so take care of that
    for i, name in enumerate(["maps", "wts", "insurance"]):
        t = i + 2
        df[f"clicks_{name}_indicator"] = np.where(
            df["treatment"] == t,
            1,
            df[f"clicks_{name}_indicator"]
            )

    # survey respondent has read all flood-related texts
    df["read_all_flood_texts"] = np.where(
        (
            df["clicks_maps_indicator"] == 1) &
            (df["clicks_wts_indicator"] == 1) &
            (df["clicks_insurance_indicator"] == 1),
        1, np.where(
            (df["clicks_maps_indicator"] == 0) |
            (df["clicks_wts_indicator"] == 0) |
            (df["clicks_insurance_indicator"] == 0),
            0, np.nan
        ))

    # add order of clicked boxes
    df = return_order_of_clicked_boxes(df)

    return df


def merge_same_question_different_treatment(df, colname):
    """Take columns named "colname", "colname.1", "colname.2", and
    "colname.3" (for whatever string `colname` is equal to) from
    Pandas.DataFrame `df`, and merge them to one unique column named
    "colname". Each row must have a well-defined value for only one
    of the "colname" columns, and a numpy nan elsewhere.

    >>> colDF = pd.DataFrame([
            [1, np.nan, np.nan, np.nan],
            [np.nan, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan],
            [np.nan, np.nan, np.nan, 1]
            ],
        columns=["c", "c.1", "c.2", "c.3"]
        )
    >>> merge_same_question_different_treatment(colDF, "c")
        c
    0  1.0
    1  1.0
    2  1.0
    3  1.0

    """
    # get columns to "merge"
    cols_to_drop = [colname, f"{colname}.1", f"{colname}.2", f"{colname}.3"]
    colsDF = df[cols_to_drop]

    # numpy.nan if nans everywhere, otherwise click time
    result = [
        np.nan if pd.isnull(list).all() else list[pd.notnull(list)][0]
        for list in colsDF.values
    ]

    # drop old columns and save new one
    df = df.drop(columns=cols_to_drop)
    df[colname] = result

    return df


def format_timestamps(df):
    """Extract datetime objects from relevant columns in Pandas.DataFrame
    `df`, i.e. those with prefix "timestamps". The latter columnes contains
    the (stringed) timestamps referring to each time each toggle box was clicked.

    For example, if the first row of column "timestamps_maps" contains the value:
    'Fri Dec 23 2022 14:18:24 GMT+0100 (CET);', this means that the first-row
    survey respondent clicked the toggle box containing information on floodmaps
    exactly one time, on Friday, December 23th 2022, at 14:18:24 Central European
    Time.

    """
    # get timestamps columns
    tscols = df.columns[df.columns.str.startswith("timestamps")]
    # harmonize missing values
    df[tscols] = df[tscols].fillna("0")

    # for each timestamp column
    for col in tscols:
        # split string of timestamps into list of timestamps
        df[col] = df[col].str.split("; ")
        # replace timestamp strings with datetime objects
        time_objects = []

        for vals in df[col]:
            # extract time (as a string)
            new_vals = [val[16:24] for val in vals]
            # convert time string to datetime object
            dates = [
                datetime.datetime.strptime(nv, "%H:%M:%S").time() for nv in new_vals if nv != ""
            ]
            # store into list
            time_objects.append(dates)

        # replace old timestamp column with new values
        df[col] = time_objects

    return df


def compute_time_spent_on_info(series):
    """Compute time spent on each toggle box containing information.

    The Qualtrics survey is programmed in such a way that three toggle
    boxes are presented simultaneously to each survey respondent, and
    only one toggle box can remain open at any given time. Any open
    toggle box will be closed if the survey respondent clicks on it
    again or clicks on another toggle box. Moreover, each toggle box
    is associated to a variable storing timestamps of each time the
    toggle box was clicked. As a result, it is possible to compute
    the time spent on each toggle box from the timestamps.

    For each survey respondents (whose timestamps for all toggle boxes
    is stored in `series`), this algorithm figures out the order in which
    the toggle boxes have been clicked, and uses it to derive the total
    number of seconds spent on each toggle box.

    """

    # extract values from series of times objects
    tbox_ts = series.values.tolist()
    # flatten the list of values
    flat_tbox_ts = [item for sublist in tbox_ts for item in sublist]
    # extract name of each value (it refers to the button that was clicked)
    tbox_names = series.index.tolist()
    durationDict = {}

    # for each time objects referring to a specific toggle box
    for i, (ts, tn) in enumerate(zip(tbox_ts, tbox_names)):
        # create a dictionary where the total seconds spend reading
        # the content of a given toggle box will be stored
        durationDict[tn] = {}
        durations = []
        next_t = 0

        # for each time object in the list of timestamps representing
        # how many times a toggle box has been clicked
        for j, item in enumerate(ts):
            # save current time object
            current_t = ts[j]

            # skip this time object if survey respondent clicked
            # on the same toggle box just before.
            # Note: the Qualtrics survey is programmed such that
            # double-clicking closes a toggle box.
            if current_t == next_t:
                pass

            else:
                # get all time objects besides the current one
                all_other_timestamps = [t for t in flat_tbox_ts if t is not current_t]
                # keep only valid time objects (respondents may have not clicked on some
                # toggle boxes, whose timestamps are then empty)
                all_other_timestamps = [
                    t for t in all_other_timestamps if current_t != ""
                ]
                # extract the time objects later than the current ones
                later_t = [
                    item for item in np.array(all_other_timestamps) if item > current_t
                ]
                # keep the time object referring to the fist toggle box that has been
                # clicked after the current one. If there is none (i.e. the current
                # time object refers to the last click... it should be "timestamps_submit"),
                # save the next time object as "end"
                next_t = min(later_t) if len(later_t) > 0 else "end"

                if next_t != "end":
                    # compute time spent on the current time object
                    duration = datetime.datetime.combine(date.min, next_t) - datetime.datetime.combine(
                        date.min, current_t
                    )
                    # store in list
                    durations.append(duration)

        # compute total seconds spent on togglebox `tn`
        total_seconds = sum(durations, timedelta()).total_seconds()
        # store into "duration dictionary"
        durationDict[tn] = total_seconds

    # convert duration dictionary into Pandas.Series
    totaltime = pd.Series(durationDict)
    # rename index of series
    totaltime.index = totaltime.index.str.replace("timestamps", "total_seconds")

    series = totaltime

    return series


def add_time_spent_on_treatment_text(df):
    """Add the time spent on the treatment text ("{text}_time_page_submit")
    to the appropriate `total_seconds_`-prefixed column for each survey
    respondent.

    I derived the time spent on each text from the toggle box,
    therefore by design the columns prefixed by `total_seconds_`
    are empty whenever the text was not in the toggle box page
    (e.g. `total_seconds_maps` is empty for survey respondents in
    treatment 2).

    """
    for i, text in enumerate(["decoy", "maps", "wts", "insurance"]):
        # for each treatment (remember that Python starts counting from 0,
        # while the treatment names go from 1 to 4)
        treatment = i+1
        # add time spent reading the treatment text wherever the column
        # prefixed by "total_seconds" is 0
        df[f"total_seconds_{text}"] = np.where(
            (df["treatment"] == treatment) & (df[f"total_seconds_{text}"] == 0) & (df["text_time_page_submit"].notna()),
            df["text_time_page_submit"],
            df[f"total_seconds_{text}"]
            )

    return df


def make_adjustment_for_pilot_data(df):
    """For some observations (less than 20, from the pilot), the timestamp of the submit button
    in the toggle boxes page is missing (my mistake, I misunderstood how the total
    time spent on the page is saved by Qualtrics. I thought Qualtrics saved timestamps).
    As a result, I need to compute the time spent on the last toggle box differently.

    """
    # total time spent on page with toggle boxes (from the first click ever on
    # the page to the click on the submit button)
    # Note: two people left the survey while on the toggle boxes page, so their
    # "info_time_page_submit" and "info_time_first_click" variables were not stored
    df["info_time_toggle"] = df["info_time_page_submit"] - df["info_time_first_click"]

    # for each toggle box
    for t in ["maps", "wts", "insurance", "decoy"]:
        # subtract the time spent on the other toggle boxes from the
        # total time spent on the page. This method is not perfectly
        # clean (the first click ever may not be on a toggle box),
        # but since this only affects one of the toggle boxes variable
        # for 18 survey participants we'll make do.
        df["approximate_time"] = (df["info_time_toggle"]
            - df["total_seconds_maps"]
            - df["total_seconds_wts"]
            - df["total_seconds_insurance"]
            - df["total_seconds_decoy"]
        )
        # one
        df["approximate_time"] = np.where(df["approximate_time"] < 0, 0, df["approximate_time"])
        df[f"total_seconds_{t}"] = np.where(
            # wherever the toggle box has been clicked, but the total seconds
            # are set to zero (because "timestamps_submit") is missing
            (df[f"clicks_{t}"] == 1) & (df[f"total_seconds_{t}"] == 0) & (df["info_time_toggle"].notna()),
            df["approximate_time"],
            df[f"total_seconds_{t}"],
        )

    return df


def return_order_of_clicked_boxes(df):
    """Add columns indicating order of clicked boxes to `df`."""

    treatment_names = ["decoy", "maps", "wts", "insurance"]
    timestamps_cols = [f"timestamps_{name}" for name in treatment_names]
    order_df = df[timestamps_cols]

    order_df = order_df.apply(return_ordered_tuple, axis=0)
    order_df = order_df.apply(return_boxes_order, axis=1).to_frame(name="clicked_boxes_order")

    df["first_box_clicked"] = order_df.clicked_boxes_order.str[0]
    df["second_box_clicked"] = order_df.clicked_boxes_order.str[1]
    df["third_box_clicked"] = order_df.clicked_boxes_order.str[2]

    return df


def return_ordered_tuple(series):
    """Return tuple of string and first element of each value
    in Pandas.Series `series`.

    """
    series = series.str[0]
    col_name = series.name.replace("timestamps_", "")
    series = [tuple([col_name, x]) for x in series]

    return series


def return_boxes_order(series):
    """Sort order of tuple of string and datetime object."""

    datetimes_to_sort = [x for x in series.values if x[1] == x[1]]
    sorted(datetimes_to_sort, key=lambda x: x[1])
    order = [x[0] for x in datetimes_to_sort]

    return order


def clean_covariates_data(df):
    """Clean answers to questions about households and survey respondent."""

    # adjust number of household members, from any number to 1, 2, 3+
    # note: I convert to float because I want to show nans, but this means
    # that I will have to rename the columns' names once I convert `household_members`
    # into categorical variables. See below
    df["household_members"] = df["household_members"].astype("float")
    df["household_members"] = np.where(
        df["household_members"] >= 3, "3+", df["household_members"]
    )

    # create dummies from categorical covariates
    categorical_covs = [
        "household_members",
        "how_long_residing",
        "homeownership",
        "julyflood_damages",
        "education",
        "household_income",
        "gender",
        "age",
        "municipality",
        "own_background",
    ]
    # It is important that all missing values are coded in the same way,
    # since I also want a categorical variable that keeps track of this
    df[categorical_covs] = df[categorical_covs].fillna("Missing value")
    df[categorical_covs] = df[categorical_covs].replace("nan", "Missing value")
    dummies = pd.get_dummies(df[categorical_covs], dtype=float)
    # clean categorical columns' names
    dummies.columns = (
        dummies.columns.str.replace(" ", "_")
        .str.replace("-", "")
        .str.replace("/", "_")
        .str.replace("'", "")
        .str.lower()
    )

    # fix little issue with the name of household members' categorical variables,
    # as discussed above
    dummies = dummies.rename(
        columns={
            "household_members_1.0": "household_members_1",
            "household_members_2.0": "household_members_2",
        }
    )

    # put everything together
    df = pd.concat([df, dummies], axis=1)

    # create derived covariates for randomization tests
    df["time_residing_m10"] = np.where(
        (df["how_long_residing_between_10_and_20_years"] == 1) |
        (df["how_long_residing_more_than_20_years"] == 1), 1, 0)

    df["time_planning_m10"] = np.where(
        (df["how_long_planning"] == "More than 20 years") |
        (df["how_long_planning"] == "Between 10 and 20 years"), 1, 0)

    df["age_younger_than_45"] = np.where(
        (df["age_between_18_and_24_years_old"] == 1) |
        (df["age_between_25_and_44_years_old"] == 1), 1, 0)

    df["household_income_min2500"] = np.where(
        (df["household_income_less_than_1200_eur"] == 1) |
        (df["household_income_between_1200_and_1700_eur"] == 1) |
        (df["household_income_between_1700_and_2500_eur"] == 1), 1, 0)

    df["household_income_m2500"] = np.where(
        (df["household_income_between_2500_and_4000_eur"] == 1) |
        (df["household_income_more_than_4000_eur"] == 1), 1, 0)

    df["edu_hbo_or_higher"] = np.where(
        (df["education_hbo"] == 1) |
        (df["education_wo"] == 1), 1, 0)

    # create dummy variables for preparedness measures
    df["preparedness_dont_know"] = np.where(df.preparedness == "Don't know", 1, 0)
    df["preparedness_no_measures"] = np.where(df.preparedness == "None of the measures above", 1, 0)
    df["preparedness_missing_value"] = np.where(df.preparedness.isna(), 1, 0)

    # only keep actually implemented measures
    df["measures"] = (df.preparedness.replace("Don't know", np.nan)
        .replace("None of the measures above", np.nan)
        .str.split(",")
    )
    # turn all values into lists (np.nans too)
    df["measures"] = [[x] if isinstance(x, list) is False else x for x in df.measures]

    measures = [
        "My household has insurance coverage against floods from extreme rain",
        "My household has insurance coverage against riverine floods",
        "My house is elevated above the street level",
        "My house walls and/or floors are built with water-resistant materials",
        "My house has anti-backflow valves installed on pipes",
        "My house has a pump and/or other systems to drain flood water installed",
        "In my house there are sandbags or other water barriers (e.g. water-proof basement windows)",
        "My valuable assets are on higher floors or elevated areas"
    ]

    measure_cols = [
        "preparedness_has_insurance_rain",
        "preparedness_has_insurance_rivers",
        "preparedness_house_elevated",
        "preparedness_house_water_resistant",
        "preparedness_house_has_valves",
        "preparedness_house_has_pump",
        "preparedness_house_has_sandbags",
        "preparedness_assets_are_high"
    ]

    for measure, measure_col in zip(measures, measure_cols):
        # create dummy variables for whether each measure is implemented
        df[measure_col] = [1 if measure in x else 0 for x in df.measures]

    # compute how many measures are implemented, preserving nans
    df["preparedness_how_many_measures"] = df[measure_cols].sum(axis=1)
    df["preparedness_how_many_measures"] = np.where(
        df["preparedness_missing_value"] == 1,
        np.nan,
        df["preparedness_how_many_measures"]
        )

    # create column for measures higher than average (both average and median are 1)
     # value is 0.9977, I can round it
    mean_measures = round(df["preparedness_how_many_measures"].mean())
    df["preparedness_high"] = np.where(
        df["preparedness_how_many_measures"] > mean_measures, 1, 0
        )

    # create dummy variables for no sources or missing value
    df["source_info_none"] = np.where(df.source_info == "I don't consult any source", 1, 0)
    df["source_info_missing_value"] = np.where(df.source_info.isna(), 1, 0)

    # only keep sources, all else becomes nans
    df["sources"] = (df.source_info.replace("I don't consult any source", np.nan).str.split(","))
    # turn all values into lists (np.nans too)
    df["sources"] = [[x] if isinstance(x, list) is False else x for x in df.sources]

    sources = [
        "Government websites",
        "Other websites",
        "My insurance company",
        "Neighbours or friends",
        "Other"
        ]
    source_cols = [
        "source_info_gov",
        "source_info_web",
        "source_info_ins",
        "source_info_friends",
        "source_info_other"
    ]

    for source, source_col in zip(sources, source_cols):
        # create dummy variables for whether each measure is implemented
        df[source_col] = [1 if source in x else 0 for x in df.sources]

    # compute how many sources, preserving nans
    df["source_info_how_many"] = df[source_cols].sum(axis=1)
    df["source_info_how_many"] = np.where(
        df["source_info_missing_value"] == 1,
        np.nan,
        df["source_info_how_many"]
        )

    # drop redudant columns
    df = df.drop(columns=["measures", "sources"])

    return df


def clean_wtp_data(df):
    """Clean data related to willingness-to-pay elicitations."""

    df = derive_wtp_for_info(df)

    # renaming
    rename_dict = {
        "extra_info_1": "get_info_1EUR",
        "extra_info_2": "get_info_1.5EUR",
        "extra_info_3": "get_info_2EUR",
        "extra_info_4": "get_info_2.5EUR",
        "extra_info_5": "get_info_3EUR",
        "extra_info_6": "get_info_3.5EUR",
        "extra_info_7": "get_info_4EUR",
        "extra_info_8": "get_info_4.5EUR",
        "extra_info_9": "get_info_5EUR",
        "extra_info_10": "get_info_5.5EUR",
        "extra_info_11": "get_info_6EUR",
        "insurance_wtp": "wtp_insurance",
        "cbond_info_time_First Click": "cbond_info_time_first_click",
        "cbond_info_time_Last Click": "cbond_info_time_last_click",
        "cbond_info_time_Page Submit": "cbond_info_time_page_submit",
        "cbond_info_time_Click Count": "cbond_info_time_click_count",
        "consumentenbond": "cbond_link_click",
    }
    df = df.rename(columns=rename_dict)

    df["wtp_answer"] = df["wtp_answer"].replace({
        "Ik wil het geld ontvangen": "I want to receive the money",
        "Ik wil de informatie ontvangen": "I want to receive the information"
    })

    # create winsorized columns for insurance
    df["wtp_insurance"] = df["wtp_insurance"].astype("float")
    df = winsorize(df, ["wtp_insurance"], 0.975)
    df = winsorize(df, ["wtp_insurance"], 0.99)

    return df


def derive_wtp_for_info(df):
    """Derive willingness to pay for information from incentivized survey task."""

    # get wtp for information
    wtp_cols = df.columns[df.columns.str.startswith("extra_info")]
    df[wtp_cols] = df[wtp_cols].replace(
        {"I want to receive the money": 0, "I want to receive the information": 1}
    )

    # check malformed columns (if I want the information for 2 EUR, I should also want the information for 1.5 EUR)
    answers_are_ordered = [
        all(earlier >= later for earlier, later in zip(seq, seq[1:]))
        for seq in df[wtp_cols].values
    ]

    # WTP is highest when `extra_info` is equal to 1 (as long as the column is not malformed), therefore I sum all the 1s
    wtp_incl_malformed = df[wtp_cols].values.sum(axis=1)
    wtpDF = pd.DataFrame(
        list(zip(wtp_incl_malformed, answers_are_ordered)), columns=["wtp", "ordered"]
    )

    # malformed WTP are saved as -99
    wtp_info = np.where(
        wtpDF["ordered"], wtpDF["wtp"], np.where(wtpDF["wtp"].isna(), wtpDF["wtp"], -99)
    )
    df["wtp_info"] = wtp_info

    # convert WTP values back to euros (as elicited in the survey)
    current_values = np.arange(1, 12, 1)
    eur_values = np.arange(1, 6.5, 0.5)
    values_dict = dict(zip(current_values, eur_values))
    df["wtp_info"] = df["wtp_info"].replace(values_dict)

    return df

def winsorize(df, columns, quantile):
    """Winsorize list of `columns` in Pandas.DataFrame
    `df` according to upper quantile `quantile`.

    Args:
        df (Pandas.DataFrame): Dataframe of interest.
        columns (list): List of column(s) in `df`.
        quantile (number): Upper quantile, must be
            between 0 and 1.

    Returns:
        Pandas.DataFrame with winsorized columns.

    """
    # get suffix for winsorized column(s)
    suffix = str(quantile)[2:] if quantile > 0 and quantile < 1 else str(quantile)
    columns_wins = [f"{c}_wins{suffix}" for c in columns]

    # add winsorized column(s) to `df`
    df[columns_wins] = np.where(
        df[columns] > df[columns].quantile(quantile),
        df[columns].quantile(quantile),
        df[columns]
    )

    return df

def add_outcomes_dummies(df):
    """Add columns to indicate (1) whether respondent has at least
    one outcome, and (2) whether a given outcome is present, to
    Pandas.DataFrame `df`. The latter needs to have columns
    "worry_RE", "risk_RE", "damages_RE", "comptot_RE", "compshare_RE",
    "wtp_info", and "wtp_insurance".

    """

    # add column to indicate whether respondent has at least one outcome
    outcomes = ["worry_RE", "risk_RE", "damages_RE", "comptot_RE", "compshare_RE", "wtp_info", "wtp_insurance"]
    # drop respondents who did not provide any outcome from index
    dfIndex = df[outcomes].dropna(axis = 0, how = 'all').index
    # assign value 1 to new column "any_outcome" for those who are not dropped
    df["any_outcome"] = np.where(df.index.isin(dfIndex), 1, 0)

    # add dummy indicating whether a given outcome is present
    outcomes_present = [f"{o}_present" for o in outcomes]
    df[outcomes_present] = np.where(df[outcomes].notna(), 1, 0)

    return df

def clean_text_evaluation(df):
    """Clean evaluation of treatment texts."""

    # create correct response to floodmaps treatment text
    # (it is individual-specific)
    conditions = [
        (df["flood_max_1_in_100_years"] == 1) & (df["flood_max_1_in_1000_years"] == 1),
        (df["flood_max_1_in_100_years"] == 0) & (df["flood_max_1_in_1000_years"] == 0),
        (df["flood_max_1_in_100_years"] == 0) & (df["flood_max_1_in_1000_years"] == 1),
        (df["flood_max_1_in_100_years"] == 1) & (df["flood_max_1_in_1000_years"] == 0),
    ]

    choices = [
        'The address floods',
        'The address does not flood',
        'The address floods only with "Small probability"',
        'The address floods only with "Medium probability"'
        ]

    correct_1 = 'Skylarks, goldfinches, and beavers'
    correct_2 = np.select(conditions, choices, "nan")
    correct_3 = 'Not necessarily, even if the damages were not insurable, not recoverable, and not preventable'
    correct_4 = 'Over 200 million euros in damages, 95% of the damage claims they received from households'

    # create columns of "attention check"
    df["text_attention_passed"] = np.where(
        (df["treatment"] == 1) & (df["text_attention"] == correct_1) |
        (df["treatment"] == 2) & (df["text_attention"] == correct_2) |
        (df["treatment"] == 3) & (df["text_attention"] == correct_3) |
        (df["treatment"] == 4) & (df["text_attention"] == correct_4),
        1, 0
    )

    return df


def add_floodmaps_indicators(df, waterdepth_column, floodrisk_column):
    """Derive indicators of flood risk and maximum water depth from
    existing columns named `waterdepth_column` and `floodrisk_column`
    in Pandas.DataFrame `df`.

    """
    # add water depth indicators
    waterdepth_dummies = pd.get_dummies(
        df[waterdepth_column],
        prefix="waterdepth_max",
        dtype=float
        )
    waterdepth_dummies.columns = (waterdepth_dummies.columns
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "", regex=False))

    # add flood indicators
    flood_dummies = pd.get_dummies(df[floodrisk_column], prefix="flood_max", dtype=float)
    flood_dummies.columns = (flood_dummies.columns
        .str.replace(" ", "_", regex=False))

    # create derived waterdepth measure for randomization tests
    # to accomodate synthetic data
    waterdepth_dummies["waterdepth_max_over_2m"] = np.where(
        (waterdepth_dummies["waterdepth_max_between_2_and_5m"] == 1) |
        (waterdepth_dummies["waterdepth_max_more_than_5m"] == 1), 1, 0)

    # concatenate into single dataframe
    df = pd.concat([df, waterdepth_dummies, flood_dummies], axis=1)

    return df


def add_revise_beliefs_variables(df):
    """Add variables providing information on expected direction of
    beliefs updating conditional on baseline information frictions
    for risk and damages.


    """
    beliefs = ["risk", "damages"]
    frictions = ["floodmaps", "waterdepth"]
    for belief, friction in zip(beliefs, frictions):

        # how the belief should be revised (preserve nans)
        conditions = [
            df[f"friction_{friction}"] > 0,
            df[f"friction_{friction}"] < 0,
            df[f"friction_{friction}"] == 0
            ]
        choices = ["downward", "upward", "none"]
        df[f"{belief}_should_revise"] = np.select(conditions, choices, "nan")
        df[f"{belief}_should_revise"] = df[f"{belief}_should_revise"].replace("nan", np.nan)

        # whether the belief was revised in the expected direction (preserve nans)
        conditions = [
            (df[f"{belief}_update"] > 0) & (df[f"{belief}_should_revise"] == "upward"),
            (df[f"{belief}_update"] < 0) & (df[f"{belief}_should_revise"] == "downward"),
            (df[f"{belief}_update"] == 0) & (df[f"{belief}_should_revise"] == "none"),
            ((df[f"{belief}_update"].isna()) | (df[f"{belief}_should_revise"].isna()))
            ]
        choices = [1, 1, 1, np.nan]
        df[f"{belief}_revise_expected"] = np.select(conditions, choices, 0)
    
        conditions = [
            ((df["friction_WTS_indicator"] == 1) | (df["friction_WTScomp_indicator"] == 1)),
            ((df["friction_WTS_indicator"].isna()) & (df["friction_WTScomp_indicator"].isna()))
            ]
        choices = [1, np.nan]
        df["compshare_should_revise"] = np.select(conditions, choices, 0)
        df["compshare_revise_expected"] = np.where(
            (df["compshare_should_revise"] == 1) & (df["compshare_update"] < 0), 1, 0)

    return df
