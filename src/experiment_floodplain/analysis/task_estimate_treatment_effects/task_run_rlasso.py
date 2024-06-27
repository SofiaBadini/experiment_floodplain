"""This script runs the R script ``run_lasso.R`` in *experiment_floodplain/analysis/R* and saves the result as .csv files to *bld/analysis*.

The the R script ``run_lasso.R`` uses the package `hdm <https://cran.r-project.org/web/packages/hdm/vignettes/hdm.pdf>`_ 
(High-Dimensional Metrics, :cite:`chernozhukov2018`) to select controls for the ATE estimationg via Lasso end Post-Lasso methods for
for high-dimensional approximately sparse models. The set of all controls is in *analysis/csv/formulas* (all the variables
with ``VARTYPE`` equal to ``PRE_COV``). Each .csv file contains the results of the estimation for a different (pre-specified) dependent variable.
The .csv files are used by the script ``task_estimate_ATE.py`` in a later step.

"""

import pytask
from importlib.metadata import version

from experiment_floodplain.config import BLD

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

produces = {}
for out in outcomes:
    produces[out] = BLD  / "analysis" / "outcomes" / "rlasso" / f"rlasso_{out}.csv"

if version('pytask-r') == '0.0.7':

    @pytask.mark.r
    @pytask.mark.depends_on(["../R/run_lasso.R", BLD / "data" / "survey_data.csv"])
    @pytask.mark.produces(produces)
    def task_run_rlasso():
        pass

else:

    @pytask.mark.r(script="../R/run_lasso.R")
    @pytask.mark.depends_on(BLD / "data" / "survey_data.csv")
    @pytask.mark.produces(produces)
    def task_run_rlasso():
        pass