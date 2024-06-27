"""This script uses the ``risk_beliefs.txt`` template and the data in ``summary_beliefs_risk.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "risk_beliefs.txt",
    "data": BLD / "analysis" / "beliefs" / "summary_beliefs_risk.csv",
}
produces = BLD / "tables" / "risk_beliefs.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):
    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0])

    # reorder columns
    data = data[[
        "true_risk_100", 
        "true_risk_1000", 
        "true_risk_10000", 
        "subj_risk_10", 
        "subj_risk_100", 
        "subj_risk_1000", 
        "subj_risk_10000", 
        "subj_risk_0"
    ]]

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abcdefgh"]

    stats = [
        "min",
        "25th quantile",
        "mean",
        "median",
        "75th quantile",
        "max",
        "SD",
        "MAD",
    ]

    kwargs = {}
    for key, stat in zip(keys, stats):
            kwargs = update(key, 8, get(data, stat))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)