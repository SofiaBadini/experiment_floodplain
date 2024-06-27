"""This script uses the ``control_beliefs.txt`` template and the data in ``summary_beliefs_t1.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "control_beliefs.txt",
    "data": BLD / "analysis" / "beliefs" / "summary_beliefs_t1.csv",
}
produces = BLD / "tables" / "control_beliefs.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):
    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0, 1])

    # separate dataframes (prior, posterior, updates)
    datapr = data["prior"][["risk", "damages_1000", "comptot", "compshare"]]
    datapo = data["posterior"][
        ["risk_RE", "damages_RE_1000", "comptot_RE", "compshare_RE"]
    ]
    dataup = data["updates"][
        [
            "risk_update",
            "damages_1000_update",
            "comptot_update",
            "compshare_update",
        ]
    ]

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keyspr = [x for x in "abcdefgh"]
    keyspo = ["a" + x for x in keyspr]
    keysup = ["b" + x for x in keyspr]

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
    # for each combination of keys and datasets (prior, posterior, updates)
    for keys, data in zip([keyspr, keyspo, keysup], [datapr, datapo, dataup]):
        # for each key (a, b, c ...) and each associated statistics
        for key, stat in zip(keys, stats):
            kwargs = update(key, 4, get(data, stat))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)