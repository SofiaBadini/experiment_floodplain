"""This script uses the ``reading.txt`` template and the data in ``reading_behavior.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "reading.txt",
    "data": BLD / "analysis" / "reading" / "reading_behavior.csv",
}
produces = BLD / "tables" / "reading.tex"


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):

    # call formatter
    fmt = UnseenFormatter()

    # open table saved as .txt file
    with open(depends_on["template"], "r") as file:
        template = file.read()

    # read data to fill table templates
    treatments = ["decoy", "maps", "wts", "insurance"]
    seconds = [f"conditional_total_seconds_{t}_wins975_mean" for t in treatments]
    clicks = [f"clicks_{t}_indicator" for t in treatments]
    indexes = seconds + clicks + ["text_attention_passed", "read_all_flood_texts"]

    data = pd.read_csv(
        depends_on["data"], 
        sep=";", 
        index_col=[0], 
        header=[0]
        ).loc[indexes].replace(0, 1).T
    
    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abcd"]

    for key, stat in zip(keys, ["decoy", "maps", "WTS", "insurance"]):
        kwargs = update(key, 4, get(data[clicks], stat))
        kwargs = update(f"s{key}", 4, [f"{x}s" for x in get(data[seconds], stat)])

    kwargs = update("t", 4, get(data.T, "text_attention_passed"))
    kwargs = update("r", 4, get(data.T, "read_all_flood_texts"))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)
    
    # save
    with open(produces, "w") as out:

        out.write(template)