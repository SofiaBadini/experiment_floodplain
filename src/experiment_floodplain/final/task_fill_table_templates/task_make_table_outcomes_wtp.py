"""This script uses the ``outcomes_wtp.txt`` template and the data in
*bld/analysis/outcomes* to create the corresponding .tex table.

"""

import pytask
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "template": SRC / "final" / "tables_templates" / "outcomes_wtp.txt",
    "unadjusted": BLD / "analysis" / "outcomes" / "outcomes_unadjusted.csv",
    "covs": BLD / "analysis" / "outcomes" / "outcomes_precovs.csv",
    "post": BLD / "analysis" / "outcomes" / "outcomes_rlasso_post.csv",
    "double": BLD / "analysis" / "outcomes" / "outcomes_rlasso_double.csv",
}
produces = BLD / "tables" / "outcomes_wtp.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):
    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    cols = [
        "wtp_insurance_wins99",
        "wtp_info"
    ]
    
    treatments = ["maps", "WTS", "insurance"]

    reskeys = ["unadjusted", "covs", "post", "double"]
    resdict = {key: depends_on[key] for key in reskeys}

    # load data
    datadict = {}
    for col in cols:
        datadict[col] = pd.concat(
            [
                pd.read_csv(
                    resdict[d], 
                    sep=";", 
                    index_col=[0], 
                    header=[0, 1]).loc[treatments][col] 
            for d in resdict
            ], 
            axis=1, 
            keys=resdict.keys()
            )
            
    # dataset of coefficients
    coef_names = ["coef", "Estimate, post-lasso", "Estimate, double selection"]
    datac = pd.concat(
        [datadict[col].loc[:, (slice(None), coef_names)] 
        for col in cols],
        axis=1, keys=cols
    )

    # dataset of standard deviations
    std_names = ["std", "Std. Error, post-lasso", "Std. Error, double selection"]
    datasd = pd.concat(
        [datadict[col].loc[:, (slice(None), std_names)] 
        for col in cols],
        axis=1, keys=cols
    )

    # dataset of p-values
    datap = pd.concat(
        [datadict[col].loc[:, (slice(None), "pvalue_fdr_tbsky")] 
        for col in cols],
        axis=1, keys=cols
    )
    
    # dataset of observations
    datan = pd.concat(
        [datadict[col].loc[:, (slice(None), "nobs")] 
        for col in cols],
        axis=1, keys=cols
    )

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abc"]

    for key, var in zip(keys, treatments):

        kwargs = update(key, 8, get(datac, var))
        kwargs = update(f"s{key}", 8, get(datasd, var))
        kwargs = update(f"p{key}", 8, get(datap, var))

    kwargs = update("n", 8, get(datan, "maps"))


    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)