"""This script uses the ``randomization.txt`` and ``randomization_contd.txt`` template 
and the data in *bld/analysis/randomization* to create the corresponding .tex table.

"""

import pytask
import pandas as pd

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.final.PYTHON.format_tables import UnseenFormatter, get, update, split_dataset

depends_on = {
    "rand": SRC / "final" / "tables_templates" / "randomization.txt",
    "contd": SRC / "final" / "tables_templates" / "randomization_contd.txt",
    "data": BLD / "analysis" / "randomization" / "randomization.csv",
}
produces = {
    "rand": BLD / "tables" / "randomization.tex",
    "contd": BLD / "tables" / "randomization_contd.tex"
}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_make_tables(depends_on, produces):
    # call formatter
    fmt = UnseenFormatter()

    # open table(s) saved as .txt file
    templates_dict = {}
    for template in ["rand", "contd"]:
        with open(depends_on[template], "r") as file:
            templates_dict[template] = file.read()

    # load data
    data = pd.read_csv(
        depends_on["data"], 
        sep=";", 
        index_col=[0], 
        header=[0, 1])

    datamean = data[["mean", "diff"]]
    datastats = data[["std", "ttest_pval"]]

    vars_rand = [
        'gender_male', 
        'own_background_yes', 
        'age_younger_than_45',
        'age_between_45_and_64_years_old', 
        'age_older_than_65_years_old',
        'edu_hbo_or_higher', 
        'risk', 
        'damages_wins975_1000', 
        'comptot',
        'compshare', 
        'worry_numeric'
    ]
    vars_contd = [
        'homeownership_yes',
        'household_income_min2500', 
        'household_income_m2500',
        'household_members_1', 
        'household_members_2', 
        'household_members_3m',
        'time_residing_m10', 
        'time_planning_m10', 
        'julyflood_damages_yes',
        'waterdepth_max_less_than_05m', 
        'waterdepth_max_over_2m',
        'flood_max_1_in_100_years', 
        'flood_max_1_in_1000_years',
        'flood_max_1_in_10000_years', 
        'FLOODED'
    ]
    
    # formatting dictionaries
    kwargs = {}

    # update dictionary with data
    keys_rand = [x for x in "abcdefghijk"]
    keys_contd = [x for x in "abcdefghijklmno"]

    for key, var in zip(keys_rand, vars_rand):

        kwargs = update(key, 10, get(datamean, var))        
        kwargs = update(f"s{key}", 10, get(datastats, var))
    
    # use updated dictionary to format the empty table template
    templates_dict["rand"] = templates_dict["rand"].format(**kwargs)

    # repeat for contd. table
    for key, var in zip(keys_contd, vars_contd):

        kwargs = update(key, 10, get(datamean, var))        
        kwargs = update(f"s{key}", 10, get(datastats, var))

    templates_dict["contd"] = templates_dict["contd"].format(**kwargs)

    # save
    for template in ["rand", "contd"]:
        with open(produces[template], "w") as out:
            out.write(templates_dict[template])