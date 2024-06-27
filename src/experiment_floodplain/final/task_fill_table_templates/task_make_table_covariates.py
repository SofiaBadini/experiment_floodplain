"""This script uses the ``covariates.txt`` template and the data in ``samples_covariates.csv`` 
(in *bld/analysis/covariates*) to create the corresponding .tex table.

"""

import pytask
import numpy as np
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update

depends_on = {
    "template": SRC / "final" / "tables_templates" / "covariates.txt",
    "data": BLD / "analysis" / "covariates" / "samples_covariates.csv",
}
produces = BLD / "tables" / "covariates.tex"


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
        index_col=[0, 1], 
        header=[0, 1]
        )

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    kwargs = update("n", 5, get(data, "N. obs", np.nan))
    kwargs = update("m", 2, get(data, "Gender", "Male"))
    kwargs = update("d", 2, get(data, "Background", "Dutch"))
    kwargs = update("a", 6, get(data, "Age", ["25-44", "45-64", "65+"]))
    kwargs = update("e", 4, get(data, "Education", ["HBO", "WO"]))
    kwargs = update("b", 2, get(data, "Born in the municipality", "Yes"))
    kwargs = update(
        "t",
        10,
        get(
            data,
            "Time resident",
            ["< 1 year", "1-5 years", "5-10 years", "10-20 years", "> 20 years"]
        ),
    )
    kwargs = update("h", 6, get(data, "Household size", ["1", "2", "3+"]))
    kwargs = update(
        "i",
        8,
        get(
            data,
            "Household income",
            ["1700-2500 EUR", "2500-4000 EUR", "> 4000 EUR", "Prefer not to say"],
        ),
    )
    kwargs = update("o", 2, get(data, "Homeowner", "Yes"))
    kwargs = update("f", 2, get(data, "Flood damages July 2021", "Yes"))
    kwargs = update(
        "r",
        15,
        get(
            data,
            "Flood risk",
            [
                "Return period: 1 in 100 years",
                "Return period: 1 in 1000 years",
                "Return period: 1 in 10000 years",
            ]
        ),
    )
    kwargs = update(
        "w",
        25,
        get(
            data,
            "Maximum water depth",
            [
                "< 0.5m",
                "Between 0.5 and 1m",
                "Between 1 and 1.5m",
                "Between 1.5 and 2m",
                "Between 2 and 5m",
            ]
        ),
    )
    kwargs = update("p", 5, get(data, "Flood risk", "In July 2021 floodplain"))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:

        out.write(template)