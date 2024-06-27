"""This script checks whether the randomization worked as expected,
for a range of pre-specified covariates stored as
``covariates_randomization.csv`` in *analysis/csv*. The prespecified 
covariates are listed in the `pre-analysis plan`_ (Section IV, Analysis, 
sub-section C, Treatment assignment, paragraph Integrity of randomization).

The results are saved as ``randomization.csv`` in *bld/analysis/randomization*.

.. _pre-analysis plan: https://osf.io/yxc3m

"""
import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.analysis.PYTHON.regressions import ttest, diff_in_mean

depends_on = {
    "survey_data": BLD / "data" / "survey_data.csv",
    "covariates": SRC / "analysis" / "csv" / "covariates_randomization.csv"
}
produces = BLD / "analysis" / "randomization" / "randomization.csv"

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_check_randomization(depends_on, produces):

    # survey respondents who provided some outcome
    survey_df = pd.read_csv(depends_on["survey_data"], sep=";").query("any_outcome == 1")

    # covariates of interest for randomization
    covariates = pd.read_csv(depends_on["covariates"], sep=";")
    covs = covariates.VARNAME

    # statsmodels is complaining
    survey_df = survey_df.rename(columns={"household_members_3+": "household_members_3m"})

    # mean
    treatments = ["T1", "T2", "T3", "T4"]
    mean_df = survey_df.groupby("treatment")[covs].mean().T.round(2)
    mean_df.columns = treatments

    # std
    std_df = survey_df.groupby("treatment")[covs].std().T.round(2)
    std_df.columns = treatments

    # t-tests
    df1, df2, df3, df4 = [x for _, x in survey_df.groupby("treatment")]
    test_names = ["T1 vs. T2", "T1 vs. T3", "T1 vs. T4", "T2 vs. T3", "T2 vs. T4", "T3 vs. T4"]
    datap = [
        ttest(df1, df2, covs),
        ttest(df1, df3, covs),
        ttest(df1, df4, covs),
        ttest(df2, df3, covs),
        ttest(df2, df4, covs),
        ttest(df3, df4, covs)
    ]
    test_df = pd.DataFrame(datap, index=test_names, columns=covs).T.round(3)

    # difference in mean
    datam = [
        diff_in_mean(df1, df2, covs),
        diff_in_mean(df1, df3, covs),
        diff_in_mean(df1, df4, covs),
        diff_in_mean(df2, df3, covs),
        diff_in_mean(df2, df4, covs),
        diff_in_mean(df3, df4, covs)
    ]
    diff_df = pd.DataFrame(datam, index=test_names, columns=covs).T.round(2)

    # everything together
    randomization_df = pd.concat(
        [mean_df, std_df, diff_df, test_df],
        axis=1,
        keys=["mean", "std", "diff", "ttest_pval"])

    # save
    randomization_df.to_csv(produces, sep=";")