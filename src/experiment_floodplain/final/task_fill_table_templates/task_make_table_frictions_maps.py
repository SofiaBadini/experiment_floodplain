"""This script uses the ``summary_beliefs.txt`` template and the data in ``summary_beliefs_all.csv`` 
(in *bld/analysis/beliefs*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "frictions_maps.txt",
    "data": BLD / "analysis" / "information" / "frictions_maps.csv",
}
produces = BLD / "tables" / "frictions_maps.tex"


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
    
    # reshape and drop column multiindex
    data = data.reset_index().pivot(index="correct", columns="stated")
    data.columns = data.columns.droplevel(0)

    # flood risk data
    fdata = data[[
        "Large probability (1 flood every 10 years)",
        "Medium probability (1 flood every 100 years)",
        "Small probability (1 flood every 1000 years)",
        "Extremely small probability (1 flood every 10,000 years)",
        "No flood risk"
    ]]

    # water depth dataframe
    wdata = data[[
        '0 cm', 
        'Up to 50 cm',
        'Between 50 cm and 1 m',
        'Between 1 and 1.5 m', 
        'Between 1.5 and 2 m',
        'Between 2 and 5 m',
        'More than 5 m'
    ]]

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    kwargs = update("a", 5, get(fdata, '1 in 100 years'))
    kwargs = update("b", 5, get(fdata, '1 in 1000 years'))
    kwargs = update("c", 5, get(fdata, '1 in 10000 years'))
    kwargs = update("d", 7, get(wdata, 'less than 0.5m'))
    kwargs = update("e", 7, get(wdata, 'between 0.5 and 1m'))
    kwargs = update("f", 7, get(wdata, 'between 1 and 1.5m'))
    kwargs = update(
        # there is a np.NaN here that we want to convert to "--"
        "g", 7, ["--" if x != x else x for x in get(wdata, 'between 1.5 and 2m')])
    kwargs = update("h", 7, get(wdata, 'between 2 and 5m'))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)
    
    # save
    with open(produces, "w") as out:

        out.write(template)