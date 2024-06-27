"""This script uses the ``summary_beliefs.txt`` template and the data in ``summary_beliefs_all.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "summary_beliefs.txt",
    "data": BLD / "analysis" / "beliefs" / "summary_beliefs_all.csv",
}
produces = BLD / "tables" / "summary_beliefs.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):

    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    data = pd.read_csv(
        depends_on["data"], 
        sep=";", 
        index_col=[0], 
        header=[0]
        )
    data = data[
        ["risk", "damages_1000", "comptot", "compshare"]
        ]

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
            kwargs = update(key, 4, get(data, stat))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)
    
    # save
    with open(produces, "w") as out:

        out.write(template)