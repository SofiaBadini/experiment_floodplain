"""This script uses the ``outcomes_frictions.txt`` template and the data in ``interaction_total_frictions.csv`` 
(in *bld/analysis/heterogeneity*) to create the corresponding .tex table.
"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "template": SRC / "final" / "tables_templates" / "outcomes_frictions.txt",
    "data": BLD / "analysis"/ "heterogeneity" / "interaction_total_frictions.csv",
}
produces = BLD / "tables" / "outcomes_frictions.tex"


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
        "risk_update_abs",
        "damages_1000_update_abs",
        "comptot_update_abs",
        "compshare_update_abs",
        "wtp_insurance_wins99",
        "wtp_info",
    ]
    data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0, 1])[cols]

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
        "total_frictions", 
        "maps:total_frictions", 
        "WTS:total_frictions",
        "insurance:total_frictions"
        ]

    for key, var in zip(keys, vars):

        kwargs = update(key, 6, get(datac, var))
        kwargs = update(f"s{key}", 6, get(datasd, var))
        kwargs = update(f"p{key}", 6, get(datap, var))

    kwargs = update("n", 6, get(datan, "maps"))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)