"""This script uses the ``outcomes_prior.txt`` template and the data in ``interaction_prior_beliefs.csv`` 
(in *bld/analysis/heterogeneity*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "template": SRC / "final" / "tables_templates" / "outcomes_prior.txt",
    "data": BLD / "analysis"/ "heterogeneity" / "interaction_prior_beliefs.csv",
}
produces = BLD / "tables" / "outcomes_prior.tex"


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
        "risk_standardized",
        "damages_1000_standardized",
        "comptot_standardized",
        "compshare_standardized",
        "prior_beliefs_zscore",
        "worry_numeric",
    ]
    original_data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0, 1])[cols]
    
    # adjust data formatting
    dfs = []
    for col in cols:
        df = original_data[col].dropna()
        df.index = df.index.str.replace(col, "prior")
        dfs.append(df)
    data = pd.concat(dfs, axis=1, keys=cols)

    # datasets of coefficients and standard deviations
    datac, datasd, datap = split_dataset(data, cols)   
    datan = data.loc[:, (slice(None), "nobs")]
    
    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abcdefg"]
    vars = [
        "maps", 
        "WTS", 
        "insurance", 
        "prior", 
        "maps:prior", 
        "WTS:prior",
        "insurance:prior"
        ]

    for key, var in zip(keys, vars):

        kwargs = update(key, 7, get(datac, var))
        kwargs = update(f"s{key}", 7, get(datasd, var))
        kwargs = update(f"p{key}", 7, get(datap, var))

    kwargs = update("n", 7, get(datan, "maps"))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)