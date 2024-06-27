"""This script produces all the figures related to prior and posterior beliefs distributions, 
confidence in beliefs, and belief updating. The figures are saved in .PNG format in *bld/figures/beliefs*.

"""
import pytask
import numpy as np
import pandas as pd
import seaborn as sns

from experiment_floodplain.config import BLD
import experiment_floodplain.final.PYTHON.visualize_beliefs as vis

sns.set_style("white")
sns.set_context("paper")
sns.set_palette("deep")

depends_on = BLD / "data" / "survey_data.csv"
fig_path = BLD / "figures" / "beliefs"
produces = {}
for i in range(1, 13):
    produces[f"beliefs{i}"] = fig_path / f"beliefs{i}.PNG"

@pytask.mark.depends_on(BLD / "data" / "survey_data.csv")
@pytask.mark.produces(produces)
def task_plot_beliefs(depends_on, produces):

    # create dictionary of figures
    figs_dict = {}
    
    # load survey data for respondents who provided some outcome
    survey_df = pd.read_csv(depends_on, sep=";").query("any_outcome == 1")
    
    # choose how to represent damages in plot
    edges = np.arange(0, 270, 10)
    labels_damages = ['[%d, %d)'%(edges[i], edges[i+1]) for i in range(len(edges)-2)]
    labels_damages.append(">250")

    # add variables for plotting
    survey_df = vis.add_variables_for_plotting(survey_df, edges, labels_damages)

    # histogram 10-year flood prob. vs. confidence
    labels_dict = vis.get_labels_dictionary(list(range(0, 101)), [0, 5, 10, 50, 100])
    fig = vis.plot_histogram_belief_vs_confidence(
        df=survey_df, 
        x="risk_for_plot", 
        y="risk_conf", 
        xlabel="Belief over 10-year flood probability",
        xticks=list(range(0, 101)), 
        bins=100, 
        labels_dict=labels_dict, 
        add_missing_values=True, 
        type="barplot")
    figs_dict["beliefs1"] = fig

    # histogram damages vs. confidence
    labels_dict_damages = dict(zip(list(range(0, 26)), labels_damages))
    fig = vis.plot_histogram_belief_vs_confidence(
        df=survey_df, 
        x="damages_for_plot", 
        y="damages_conf", 
        xlabel="Belief over damages (kEUR)",
        xticks=list(range(0, 26)), 
        bins=26, 
        labels_dict=labels_dict_damages, 
        add_missing_values=False, 
        rotation=45,
        type="barplot")
    figs_dict["beliefs2"] = fig

    # histogram total compensation vs. confidence
    labels_dict = vis.get_labels_dictionary(list(range(0, 101)), [0, 25, 50, 75, 100])
    fig = vis.plot_histogram_belief_vs_confidence(
        df=survey_df, 
        x="comptot", 
        y="comptot_conf", 
        xlabel="Belief over total compensation",
        xticks=list(range(0, 101)), 
        bins=100, 
        labels_dict=labels_dict, 
        add_missing_values=True, 
        type="barplot")
    figs_dict["beliefs3"] = fig
    
    # histogram total compensation vs. confidence
    fig = vis.plot_histogram_belief_vs_confidence(
        df=survey_df, 
        x="compshare", 
        y="compshare_conf", 
        xlabel="Belief over government compensation",
        xticks=list(range(0, 101)), 
        bins=100, 
        labels_dict=labels_dict, 
        add_missing_values=True, 
        type="barplot")
    figs_dict["beliefs4"] = fig


    # belief updating by direction
    query_strings_t1 = [
        # spontaneously avoid risk info
        "treatment == 1 and clicks_maps_indicator == 0",
        # spontaneously saw risk info
        "treatment == 1 and clicks_maps_indicator == 1",
        # spontaneously read risk info (time spent >= 45 seconds)
        "treatment == 1 and total_seconds_maps >= 45",
    ]

    titles_t1 = [
        "Spontaneously avoided risk information",
        "Spontaneously clicked on risk information",
        "Spontaneously read risk information",
    ]

    query_strings_t2 = [
        # was shown risk info
        "treatment == 2",
        # was shown risk info and read it (time spent >= 45 seconds)
        "treatment == 2 and total_seconds_maps >= 45",
    ]

    titles_t2 = [
        "Was shown risk information",
        "Was shown risk information and read it"
    ]

    xlabel = "Belief update over flood probability (absolute size)"
    ylabel = "Percent of respondents"
    
    fig = vis.plot_updates(survey_df, query_strings_t1, titles_t1, xlabel, ylabel, figsize=(8.25, 6))
    figs_dict["beliefs5"] = fig

    fig = vis.plot_updates(survey_df, query_strings_t2, titles_t2, xlabel, ylabel, figsize=(8.25, 4))
    figs_dict["beliefs6"] = fig

    # heatmap
    _temp_dict = {
        "risk": "10-year flood prob.", 
        "damages_wins975_1000": "Damages", 
        "comptot": "Tot. comp.", 
        "compshare": "Govt. comp.", 
        "worry_numeric": "Worry"
    }
    df_corr = survey_df.rename(columns=_temp_dict)[_temp_dict.values()].corr()
    fig = vis.plot_lower_triangular_heatmap(
        df_corr, 
        "", 
        884,
        "Blues"
        )
    figs_dict["beliefs7"] = fig

    # plot histograms vs. scatterplots
    priors = ["risk", "damages_wins975_1000", "comptot", "compshare", "worry_numeric"]
    posteriors = ["risk_RE", "damages_RE_wins975_1000", "comptot_RE", "compshare_RE", "worry_RE_numeric"]
    xlabels = [
        "10-year flood prob.", 
        "Damages (kEUR)", 
        "Tot. comp. (%)", 
        "Govt. comp. (%)",
        "Worry"]
    
    for i, (prior, posterior, xlabel) in enumerate(zip(priors, posteriors, xlabels)):
        
        fig = vis.histogram_scatterplot_belief_distributions(
            survey_df.query("treatment == 1"), x=prior, y=posterior, xlabel=xlabel)
        
        n = 8 + i

        figs_dict[f"beliefs{n}"] = fig

    
    # save all the figures
    for key in produces.keys():
        figs_dict[key].savefig(
            produces[key], dpi=350, bbox_inches="tight"
        )