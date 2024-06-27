"""This script produces the figures related to survey respondents' elicited
willingness-to-pay for insurance. The figures are saved 
in .PNG format in *bld/figures/willingness_to_pay*.

"""

import pytask
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl 

from experiment_floodplain.config import BLD
import experiment_floodplain.final.PYTHON.visualize_wtp as vis

sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")

mpl.rcParams['hatch.linewidth'] = 0.65

depends_on = BLD / "data" / "survey_data.csv"
fig_path = BLD / "figures" / "wtp"
produces = {"wtp1": fig_path / "wtp1.PNG", "wtp2": fig_path / "wtp2.PNG"}

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_plot_wtp(depends_on, produces):

    # create dictionary of figures
    figs_dict = {}

    # load survey data for respondents who provided some outcome
    survey_df = pd.read_csv(depends_on, sep=";").query("any_outcome == 1")

    # add temporary columns for plotting
    survey_df["is_wtp_ins_positive"] = np.where(
        survey_df["wtp_insurance"] > 0, 1, np.where(
            survey_df["wtp_insurance"] <= 0, 0, np.nan
        ))
    survey_df["wtp_ins_strictly_positive"] = np.where(
        survey_df["wtp_insurance_wins975"] > 0, 
        survey_df["wtp_insurance_wins975"], 
        np.nan
        )

    # plots    
    data = survey_df[["is_wtp_ins_positive", "wtp_insurance_wins975", "correct_floodmaps", "treatment"]].dropna()
    fig = vis.plot_insurance_two_arms(data)
    fig.savefig(produces["wtp1"], bbox_inches="tight", dpi=350)

    sns.set_context("notebook") # it just looks better, what can I say...
    fig = vis.plot_wtp_distributions(survey_df)
    fig.savefig(produces["wtp2"], dpi=350, bbox_inches="tight")