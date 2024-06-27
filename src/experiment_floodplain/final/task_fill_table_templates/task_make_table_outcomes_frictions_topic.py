"""This script uses the ``outcomes_frictions_topic.txt`` template and the data in ``interaction_frictions_by_topic.csv``
(in *bld/analysis/heterogeneity*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "template": SRC / "final" / "tables_templates" / "outcomes_frictions_topic.txt",
    "data": BLD / "analysis"/ "heterogeneity" / "interaction_frictions_by_topic.csv",
}
produces = BLD / "tables" / "outcomes_frictions_topic.tex"


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
        "risk_update",
        "damages_1000_update",
        "comptot_update",
        "compshare_update",
    ]
    data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0, 1])[cols]
    frictions = ["_floodmaps", "_waterdepth", "_comp", "_comp"]
    new_dfs = []
    for col, friction in zip(cols, frictions):
        new_df = data[col].dropna()
        old_index = new_df.index
        new_index = [_.replace(friction, "") for _ in old_index]
        new_df.index = new_index
        new_dfs.append(new_df)

    data = pd.concat(new_dfs, axis=1, keys=cols)
    
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
        "friction", 
        "maps:friction", 
        "WTS:friction",
        "insurance:friction"
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