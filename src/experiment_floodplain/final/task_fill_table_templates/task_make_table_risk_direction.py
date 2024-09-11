"""This script uses the ``risk_direction.txt`` template and the data in ``summary_risk_updates_direction.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import itertools
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "risk_direction.txt",
    "data": BLD / "analysis" / "beliefs" / "summary_risk_updates_direction.csv",
}
produces = BLD / "tables" / "risk_direction.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):
    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    index_col = ["risicokaart_flood_probability", "update_direction"]
    data = (pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0])
        .reset_index()
        .set_index(index_col)
        .round(1)
        .fillna("")
    )

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abcdefghilmn"]

    index1 = ["1 in 100 years", "1 in 1000 years", "1 in 10000 years"]
    index2 = ["no_update", "expected_direction", "unexpected_direction", "not_reported"]

    kwargs = {}
    for key, stat in zip(keys, itertools.product(index1, index2)):
        
        kwargs = update(key, 16, get(data, stat[0], stat[1]))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)