"""This script estimates the Average Treatment Effects of the intervention on 
the (pre-specified) measures of beliefs updating and worry about flood risk, as well
as the (pre-specified) measures of willigness-to-pay for insurance (hypothetical) and 
for information about Dutch insurance companies that offer protection against flood risk 
(incentivized). 

I use (1) an unadjusted linear regression estimated via OLS (``outcomes_unadjusted.csv``), 
(2) a linear regression estimated via OLS that includes a small set of pre-specified covariates
(``outcomes_precovs.csv``), and (3) two partially linear models that include a broader set of 
covariates where the nuisance functions are estimated via Lasso (``outcomes_rlasso_post.csv`` and 
``outcomes_rlasso_double.csv``). I adjust the p-values for multiple hypotheses testing using 
the two-stage Benjamini, Krieger, and Yekutieli procedure for controlling the false discovery rate (FDR)
as implemented in the Python package `statsmodels <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_
(option "fdr_tbsky").
The four dataframes of results are saved as .csv files in *bld/analysis/outcomes*. 

I additionally test whether survey respondents have a different probability to answer
any given outcome-related question across treatment arms, via a logistic regression 
where the dependent variable indicates whether such outcome is present in the data.
The results for it are saved as ``whether_outcome_present.csv``,
also in *bld/analysis/outcomes*.

"""
import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.regressions import (
    logit_treatment_on_response, 
    run_wls_regressions,
    adjust_pvalues,
    load_rlasso_datasets
)

rlasso_path = BLD / "analysis" / "outcomes" / "rlasso" 
outcomes = [
    "risk_update",
    "damages_1000_update",
    "damages_wins99_1000_update",
    "damages_wins975_1000_update",
    "compshare_update",
    "comptot_update",
    "worry_numeric_update",
    "wtp_insurance",
    "wtp_insurance_wins99",
    "wtp_insurance_wins975",
    "wtp_info"
]
files = [rlasso_path / f"rlasso_{outcome}.csv" for outcome in outcomes]
rlasso_dict = dict(zip(outcomes, files))

# rlasso outcomes
depends_on = rlasso_dict
depends_on.update({"survey_data": BLD / "data" / "survey_data.csv"})
depends_on.update({"formulas": SRC / "analysis" / "csv" / "formulas.csv"})

outcomes_path = BLD / "analysis"/ "outcomes"
produces = {
    "whether_outcome_present": outcomes_path / "whether_outcome_present.csv",
    "outcomes_unadjusted": outcomes_path / "outcomes_unadjusted.csv",
    "outcomes_precovs": outcomes_path / "outcomes_precovs.csv",
    "outcomes_rlasso_post": outcomes_path / "outcomes_rlasso_post.csv",
    "outcomes_rlasso_double": outcomes_path / "outcomes_rlasso_double.csv",
    "wtp_no_wins": outcomes_path / "wtp_no_wins.csv",
    "wtp_wins975": outcomes_path / "wtp_wins975.csv"
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_estimate_ATE(depends_on, produces):

    # load data
    survey_df = pd.read_csv(depends_on["survey_data"], sep=";").query("any_outcome == 1")
    # convert malformed WTP to np.nan, so they are dropped from the analysis
    survey_df["wtp_info"] = np.where(survey_df["wtp_info"] == -99, np.nan, survey_df["wtp_info"])

    # only one observation with waterdepth higher than 5m
    # adjust this as we use `correct_waterdepth` as control
    survey_df.correct_waterdepth = survey_df.correct_waterdepth.replace(
        {"between 2 and 5m": "more than 2m", "more than 5m": "more than 2m"}
        )

    # check whether treatments affect if respondents answer outcomes questions
    outcomes = ["worry_RE", "risk_RE", "damages_RE", "comptot_RE", "compshare_RE", "wtp_info", "wtp_insurance"] 
    outcomes_present = [f"{o}_present" for o in outcomes]
    whether_outcome_present = pd.concat(
        [logit_treatment_on_response(survey_df, outcome, "C(treatment)") for outcome in outcomes_present],
        keys=outcomes, 
        axis=1
        )

    # extract formulas and outcomes
    formulas_df = pd.read_csv(depends_on["formulas"], sep=";")
    outcomes = ["OUTCOME_UPDATE", "OUTCOME_WTP"]
    outcomes = formulas_df.query("VARTYPE in @outcomes").VARNAME.tolist()
    
    outcomes_to_be_adjusted = [
        "risk_update", 
        "damages_1000_update",
        "comptot_update", 
        "compshare_update", 
        "worry_numeric_update", 
        "wtp_insurance_wins99", 
        "wtp_info"
        ]    
    treatments = ["maps", "WTS", "insurance"]

    # run unadjusted regression on outcomes
    depvar = "C(treatment)"
    outcomes_unadjusted = run_wls_regressions(survey_df, outcomes, depvar)
    outcomes_unadjusted_adj = adjust_pvalues(outcomes_unadjusted, treatments, outcomes_to_be_adjusted)

    # run regressions controlling for pre-specified narrow set of controls
    precovs = formulas_df.query(f"VARTYPE == 'PRE_COV'").VARNAME.tolist()
    precovs = ["C(treatment)"] + precovs
    precovs = " + ".join(precovs)
    outcomes_precovs = run_wls_regressions(survey_df, outcomes, precovs)
    outcomes_precovs_adj = adjust_pvalues(outcomes_precovs, treatments, outcomes_to_be_adjusted)

    # load rlasso outcomes 
    depends_on.pop("survey_data")
    depends_on.pop("formulas")
    outcomes_rlasso_post, outcomes_rlasso_double = load_rlasso_datasets(depends_on)
    
    # add number of observations to rlasso outcomes
    covs_extended = formulas_df.query(f"VARTYPE == 'PRE_COV_EXTENDED'").VARNAME.tolist()
    for out in outcomes:
        all_vars = [out] + covs_extended
        nobs = len(survey_df[all_vars].dropna())
        outcomes_rlasso_post.loc[:, (out, "nobs")] = nobs
        outcomes_rlasso_double.loc[:, (out, "nobs")] = nobs

    # adjust p-values for rlasso outcomes
    outcomes_rlasso_post_adj = adjust_pvalues(
        outcomes_rlasso_post, 
        treatments, 
        outcomes_to_be_adjusted, 
        pval_col=f"Pr(>|t|), post-lasso")
        
    outcomes_rlasso_double_adj = adjust_pvalues(
        outcomes_rlasso_double, 
        treatments, 
        outcomes_to_be_adjusted, 
        pval_col=f"Pr(>|t|), double selection")
    
    # add table with alternative winsorization of WTP for insurance

    res_dict = {}

    for wtp in ["wtp_insurance", "wtp_insurance_wins975"]:
    
        outcomes_wins = [
            outcome 
            if outcome != 'wtp_insurance_wins99' 
            else wtp
            for outcome 
            in outcomes_to_be_adjusted 
            ]
    
        wtp_unadjusted = adjust_pvalues(outcomes_unadjusted, treatments, outcomes_wins)[wtp]
        wtp_precovs = adjust_pvalues(outcomes_precovs, treatments, outcomes_wins)[wtp]
        wtp_rlasso_post = adjust_pvalues(
            outcomes_rlasso_post, treatments, outcomes_wins, pval_col=f"Pr(>|t|), post-lasso")[wtp]
        wtp_rlasso_double = adjust_pvalues(
            outcomes_rlasso_double, treatments, outcomes_wins, pval_col=f"Pr(>|t|), double selection")[wtp]
    
        res_dict[wtp] = pd.concat(
            [wtp_unadjusted, wtp_precovs, wtp_rlasso_post, wtp_rlasso_double], 
            axis=1, 
            keys=["unadjusted", "wtp_precovs", "wtp_rlasso_post", "wtp_rlasso_double"])
    
    # save
    whether_outcome_present.round(2).to_csv(produces["whether_outcome_present"], sep=";")
    outcomes_unadjusted_adj.round(2).to_csv(produces["outcomes_unadjusted"], sep=";")
    outcomes_precovs_adj.round(2).to_csv(produces["outcomes_precovs"], sep=";")
    outcomes_rlasso_post_adj.round(2).to_csv(produces["outcomes_rlasso_post"], sep=";")
    outcomes_rlasso_double_adj.round(2).to_csv(produces["outcomes_rlasso_double"], sep=";")
    res_dict["wtp_insurance"].round(2).to_csv(produces["wtp_no_wins"], sep=";")
    res_dict["wtp_insurance_wins975"].round(2).to_csv(produces["wtp_wins975"], sep=";")