"""Functions shared across the two modules performing checks for the integrity of the experimental 
randomization, and estimation of the treatment effects.

"""
import re
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pathlib import Path
from statsmodels.stats.multitest import multipletests


def load_rlasso_datasets(path_dict):
    """Load datasets containing outcomes of Rlasso regressions."""
    rlasso_dict = {}

    for outcome_name, path in path_dict.items():
        df = pd.read_csv(path, index_col=[0]).T
        rlasso_dict[outcome_name] = df

    outcomes_rlasso = pd.concat(
        list(rlasso_dict.values()),
        axis=1,
        keys=rlasso_dict.keys()
        )

    stats = ['Estimate', 'Std. Error', 't value', 'Pr(>|t|)']
    stats_post = [f"{s}, post-lasso" for s in stats]
    stats_double = [f"{s}, double selection" for s in stats]

    outcomes_rlasso_post =  outcomes_rlasso.loc[:, (slice(None), stats_post)]
    outcomes_rlasso_double =  outcomes_rlasso.loc[:, (slice(None), stats_double)]

    return outcomes_rlasso_post, outcomes_rlasso_double


def ttest(df1, df2, covs):
    """Compute t-test for difference in means along `covs` between `df1` and `df2`."""
    pvals = [scipy.stats.ttest_ind(df1[cov].dropna(), df2[cov].dropna())[1] for cov in covs]
    return pvals


def diff_in_mean(df1, df2, covs):
    """Compute difference in means along `covs` between `df1` and `df2`."""
    diff_in_mean = [df1[cov].dropna().mean() - df2[cov].dropna().mean() for cov in covs]
    return diff_in_mean


def logit_treatment_on_response(df, outcome, dep_vars):
    """Logistic regression where the dependent
    variable is `outcome` and the independent
    variable(s) `dep_vars` need to contain variable
    "C(treatment)".

    """

    # logistic regression
    mod = smf.logit(
        data=df,
        formula=f'{outcome} ~ {dep_vars}'
    )
    res = mod.fit(cov_type="HC1")
    res_df = (res.summary2().tables[1])

    # add number of observations and LLR p-value
    res_df["nobs"] = res.nobs
    res_df["llr_pvalue"] = res.llr_pvalue
    res_df = res_df.rename(
    columns={"Coef.": "coef", "Std.Err.": "std", "P>|t|" : "pval"},
        index={
            "C(treatment)[T.2.0]": "maps",
            "C(treatment)[T.3.0]": "WTS",
            "C(treatment)[T.4.0]": "insurance",
            }
    )
    # round to 2 decimal places for legibility
    res_df = res_df.round(2)

    return res_df


def run_wls_regressions(data, outcomes, depvar):
    """Run werighted least square regressions of 
    `depvar` on `outcomes`, where both are columns of
    the Pandas.DataFrame `data`.
    
    """

    outcomes_df = (pd.concat(
    [weighted_least_squares_regression(data, outcome, depvar) 
    for outcome in outcomes],
    axis=1, keys=outcomes).fillna("")
    )

    return outcomes_df


def weighted_least_squares_regression(df, outcome, dep_vars):
    """Run weighted least squares regression on Pandas.DataFrame
    `df`, according to `formula`. `df needs to contain columns
    "WEIGHTS" and "treatment".

    Args:
        df (Pandas.DataFrame): Dataframe of interest.
        outcome (str): Outcome of formula.
        dep_vars (str): Dependent variables of linear model.

    Returns:
        Pandas.DataFrame

    """
    mod = smf.wls(
        data=df,
        formula=f"{outcome} ~ {dep_vars}",
        weights=df.WEIGHTS
    )
    res = mod.fit(cov_type="HC1")
    print(res.summary())
    res_df = (res.summary2().tables[1])

    # add number of observations and adjusted R squared
    res_df["nobs"] = res.nobs
    res_df["adj_R"] = res.rsquared_adj

    # add average outcome in treatment 1
    res_df["decoy_outcome"] = df.query("treatment == 1")[outcome].mean()

    # rename columns
    res_df = res_df.rename(
        columns={"Coef.": "coef", "Std.Err.": "std", "P>|t|" : "pval"},
        )

    # rename index
    new_index = res_df.index.tolist()
    for t, tname in zip([2, 3, 4], ["maps", "WTS", "insurance"]):
        new_index = [i.replace(f"C(treatment)[T.{t}.0]", tname) for i in new_index]

    res_df.index = new_index

    # round to 2 decimals for legibility
    res_df = res_df

    return res_df


def create_dictionary_of_covariates(
        formulas_df, outcomes, add_baseline_beliefs=True, add_het=False, het_covs=list(), covs_to_remove=list()):
    """Create dictionary relating each treatment outcome to its set of covariates.

    Args:
        formulas_df (Pandas.DataFrame): Dataframe of covariates to use.
        outcomes (list): List of outcomes.
        add_baseline_beliefs (boolean): Add outcome-specific baseline beliefs and confidence
            in beliefs as covariates. Default is True.
        add_het (boolean): Whether to add variables for heterogeneity analysis to the list
            of covariates. If True, `het_covs` need to be a dictionary or a list of variables.
            Default is False.
        het_covs (list or dictionary): List of variables for heterogeneity analysis or, if the
            heterogeneity analysis is outcome-specific, dictionary where keys are outcomes and
            values are list of variables. Default is empty list.
        covs_to_remove (list or dictionary): List of variables to be removed from the list of
            covariates, or dictionary if the variables to remove are outcome-specific.
            Default is empty list.

    Returns:
        dictionary.

    """
    covs_dict = {}

    for outcome in outcomes:

        # select general pre-specified covariates
        covs_small = formulas_df.query(f"VARTYPE == 'PRE_COV'").VARNAME.tolist()
        all_covs = ["C(treatment)"] + covs_small

        # select outcome-specific pre-specified covariates
        if add_baseline_beliefs:

            extra_covs = formulas_df.query(f"VARTYPE == 'BASELINE_BELIEFS_{outcome}'").VARNAME.tolist()
            all_covs = all_covs + extra_covs

        if add_het:

            # extract heterogeneity analysis covariates
            _het_covs = het_covs[outcome] if isinstance(het_covs, dict) else het_covs
            _het_covs = [_het_covs] if isinstance(_het_covs, str) else _het_covs

            # extract covariates to remove and remove them to all covariates
            _covs_to_remove = covs_to_remove[outcome] if isinstance(covs_to_remove, dict) else covs_to_remove
            all_covs = [cov for cov in all_covs if cov not in _covs_to_remove]

            all_covs = all_covs + _het_covs

            # remove duplicates if there are any
            all_covs = list(set(all_covs))

        # add to dictionary as formula
        covs_dict[outcome] = " + ".join(all_covs)

    return covs_dict


def adjust_pvalues(df, treatments, outcomes, pval_col="P>|z|"):
    """Adjust the p-values in `df` using the two-stage
    Benjamini, Krieger, and Yekutieli procedure for controlling
    the false discovery rate (FDR). Add the adjusted p-values
    as a new column named "pvalue_fdr_tbsky" to `df`.

    Args:
        df (Pandas.DataFrame): Dataframe of interest. Need to have
            the column "P>|z|".
        treatments (list of strings): Index names to subset.
            Need to be three.
        outcomes (list of strings): Column names to subset.
            Need to be seven.
        pval_col (str): Name of p-value column. Default is P>|z|.

    Returns:
        Pandas.DataFrame

    """
    pvals = df.loc[treatments, (outcomes, pval_col)].values.flatten()

    pvals_adjusted = multipletests(
        pvals,
        alpha=0.05,
        method='fdr_tsbky',
        maxiter=1,
        is_sorted=False,
        returnsorted=False)[1]

    # re-create dictionary of pvalues with initial shape
    pvals_df = pd.DataFrame(
        [pvals_adjusted[:7], pvals_adjusted[7:14], pvals_adjusted[14:21]],
        columns=pd.MultiIndex.from_product([outcomes, ["pvalue_fdr_tbsky"]]),
        index=treatments
        )

    df_updated = pd.concat([df, pvals_df], axis=1)

    return df_updated


def wls_on_beliefs_percentile(
        df, belief, outcome, covariates, percentile, only_keep_interactions=True):
    """Run weighted least squares with interaction effect over treatment
    and percentile of prior beliefs.

    Args:
        df (Pandas.DataFrame): Dataframe of interest.
        belief (str): Belief of interest.
        outcome (str): Outcome of interest.
        covariates (list of str): Covariates for wls regression.
        percentile (str): "quartile" or "tercile".
        only_keep_interactions (bool): Whether to only store interactions
            coefficients in the dataframe of results. Default is True.

    Returns:
        Pandas.DataFrame

    """

    het_var = [f"C(treatment)*C({belief}_{percentile})"]
    covariates_beliefs = het_var + covariates
    covariates_str = " + ".join(covariates_beliefs)

    p1 = 'q1' if percentile == 'quartile' else 't1'
    p = 'q' if percentile == 'quartile' else 't'

    belief_df = weighted_least_squares_regression(
        df, outcome, covariates_str
        )
    outcome_string = f"treatment == 1 and {belief}_{percentile} == @p1"
    belief_df["decoy_outcome"] = (df.query(outcome_string)[outcome].mean())
    belief_df = belief_df.rename(columns={"decoy_outcome": f"t1_{p1}_outcome"})

    for n in [2, 3, 4]:
        # rename interaction effects for legibility
        belief_df.index = belief_df.index.str.replace(
            f"C({belief}_{percentile})[T.{p}{n}]",
            f"prior_{p}{n}"
            )

    if only_keep_interactions:
        # only keep rows with interactions effect
        treatments = ["maps", "WTS", "insurance"]
        interactions = belief_df.index[belief_df.index.str.contains(f"prior_{p}")].tolist()
        index_to_keep = treatments + interactions

        belief_df = belief_df.loc[index_to_keep]

    return belief_df


def wls_with_interaction(df, outcome, covariates, interaction, keep_only_interaction=True, var_to_add=None):
    """Run weighted least squares with interacted treatment.

    Args:
        df (Pandas.DataFrame): Dataframe from which variables are extracted.
        outcome (str): Name of outcome variable.
        covariates (list of strings): Names of covariates.
        interaction (str): Name of interaction variable.
        keep_only_interaction (boolean): Keep only estimated coefficient of
            treatment and interacted treatment. Default is True.

    Returns:
        Pandas.DataFrame

    """
    covariates = [c for c in covariates if c != interaction]
    regressors = [f"C(treatment)*{interaction}"] + covariates
    regressors_str = " + ".join(regressors)
    res_df = weighted_least_squares_regression(df, outcome, regressors_str)
    res_df = res_df.drop(columns=["decoy_outcome"])

    if keep_only_interaction:
        interaction_index = res_df.index[res_df.index.str.contains(interaction)].tolist()
        index_to_keep = ["maps", "WTS", "insurance"] + interaction_index
        if var_to_add:
            index_to_keep = index_to_keep + [var_to_add]
        res_df = res_df.loc[index_to_keep]

    return res_df