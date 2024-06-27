"""This script uses the ``outcomes_present.txt`` template and the data in ``whether_outcome_present.csv`` 
(in *bld/analysis/outcomes*) to create the corresponding .tex table.

"""

import pytask
import pandas as pd

from experiment_floodplain.config import SRC, BLD

from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "template": SRC / "final" / "tables_templates" / "outcomes_present.txt",
    "data": BLD / "analysis" / "outcomes" / "whether_outcome_present.csv",
}
produces = BLD / "tables" / "outcomes_present.tex"


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
        "risk_RE",
        "damages_RE",
        "comptot_RE",
        "compshare_RE",
        "worry_RE",
        "wtp_insurance",
        "wtp_info",
    ]
    data = pd.read_csv(depends_on["data"], sep=";", index_col=[0], header=[0, 1])[cols]

    # datasets of coefficients, standard deviations and p-value
    datac, datasd, datap = split_dataset(data, cols)
    datan = data.loc[:, (slice(None), "nobs")]
    datallr = data.loc[:, (slice(None), "llr_pvalue")]

    # formatting dictionary
    kwargs = {}

    # update dictionary with data
    keys = [x for x in "abcd"]
    vars = [
        "Intercept", 
        "maps", 
        "WTS", 
        "insurance"
        ]

    for key, var in zip(keys, vars):

        kwargs = update(key, 7, get(datac, var))
        kwargs = update(f"s{key}", 7, get(datasd, var))
        kwargs = update(f"q{key}", 7, get(datap, var))

    kwargs = update("n", 7, get(datan, "Intercept"))
    kwargs = update("p", 7, get(datallr, "Intercept"))

    # use updated dictionary to format the empty table template
    template = template.format(**kwargs)

    # save
    with open(produces, "w") as out:
        out.write(template)