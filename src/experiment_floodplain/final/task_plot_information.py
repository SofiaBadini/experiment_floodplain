"""This script produces all the figure related to survey respondents' information quality.
The figures are saved in .PNG format in *bld/figures/information*.

"""

import pytask
import pandas as pd
import seaborn as sns

from experiment_floodplain.config import BLD
import experiment_floodplain.final.PYTHON.visualize_information as vis

sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")
depends_on = BLD / "data" / "survey_data.csv"
produces = {
    "information1": BLD / "figures" / "information" / "information1.PNG",
    "information2": BLD / "figures" / "information" / "information2.PNG",
    }

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_plot_information(depends_on, produces):

    # load survey data and drop survey respondents who did not provide any outcome
    survey_df = pd.read_csv(depends_on, sep=";").query("any_outcome == 1")

    # information vs. confidence
    fig = vis.plot_total_information_vs_confidence(survey_df)
    fig.savefig(produces["information1"], dpi=350,  bbox_inches='tight')

    # disaggregated information
    fig = vis.plot_information_vs_confidence(survey_df, noise=0.025)
    fig.savefig(produces["information2"], dpi=350,  bbox_inches='tight')