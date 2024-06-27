"""Visualization module for ``task_plot_information.py``."""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_total_information_vs_confidence(df):
    """Plot two histograms next two each other.
    
    The first histogram has number of information 
    frictions on the x-axis and share of survey respondets
    on the y-axis.

    The second histogram has average confidence in
    answers to information-based questions on the x-axis
    and share of survey respondents on the y-axis.
    
    The Pandas.DataFrame `df` needs to contain the columns
    "total_frictions" and "average_info_confidence".
    
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    axs = axs.flatten()

    # columns
    cols = ["total_frictions", "average_info_confidence"]

    # labels
    labels = ["Total incorrect responses", "Average confidence"]

    for ax, col, label, bins in zip(axs, cols, labels, [8, 10]):
        sns.histplot(
            ax=ax, 
            data=df, 
            x=col, 
            bins=bins, 
            discrete=True,
            stat='probability'
            )
        ax.set_xlabel(label, fontweight="bold", labelpad=10)
        sns.despine()
    
        # annotate plots with mean, median and SD
        mean = df[col].mean().round(2)
        median = df[col].median().round(2)
        SD = df[col].std().round(2)
        ax.annotate(
            f"Mean: {mean} \nMedian: {median} \nSD: {SD}", 
                xycoords="axes fraction",
                xy=(0.8, 0.9),
                fontsize=8.5,
                bbox=dict(facecolor='none', edgecolor='lightgrey', linewidth=0.5, pad=5.0))

    # set y-axis label
    axs[0].set_ylabel("% of respondents", labelpad=10)

    # set x-axis ticks for 1, 2 ... 10
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    return fig


def scatter_dataframe(df, cols, increment):
    """Scatter valued of `cols` in `df`, by increment.
    
     Args:
        df (Pandas.DataFrame): Dataset.
        cols (list of strings): Column(s) of `df` whose values should be scattered.
        increment (float): By how much should the values of `cols` be scattered.
    
    Returns:
        Pandas.DataFrame.
    
    """
    # group dataframe by columns
    grouped_object = df.groupby(cols)
    # extract group names
    group_names = grouped_object.groups.keys()
    groups = []
    
    for group_name in group_names:
        # get each group (identified by the `cols`)
        group = grouped_object.get_group(group_name).copy()
        # how many observations in each group
        n_obs = len(group)
        # observations are scattered in groups of `n_rows`
        n_rows = int(math.ceil(np.sqrt(n_obs)))
                
        # import math
        new_vals = [i for i in range(0, n_rows)] * math.ceil(n_obs / n_rows)
        group["grouped_obs"] = new_vals[:n_obs]
        
        for col in cols:
            # scatter values
            #group[col] = scatter_values(group, col, n_rows, increment)
            group[col] = scatter_values(group, col, "grouped_obs", increment)
            group = group.sort_values(by=[col])
            group["grouped_obs"] = group[col].values
        
        groups.append(group)
    
    scattered_df = pd.concat(groups)
    
    return scattered_df


def scatter_values(group, col, grouping_col, increment):
    
    group_value = group[col].values[0]
    grouped_observations = group.groupby(grouping_col)[col]
    grouped_observations_names = grouped_observations.groups.keys()

    new_vals_for_group = []
    
    for name in grouped_observations_names:
    
        current_group = grouped_observations.get_group(name)
    
        new_vals = []
        for x in range(0, len(current_group)):
            new_vals += [group_value + x*increment, group_value - x*increment]

        new_vals = new_vals[1:len(current_group)+1]
        new_vals_for_group += new_vals
    
    return new_vals_for_group


def get_df_for_plotting(df, noise):
    """Melt dataframe `df` so that each answer to an 
    information-based question is classified as correct
    or incorrect (value 1 or 0), by topic (flood maps,
    insurance, government compensation), and by confidence
    in the answer (number from 1 to 10).
    
    Args:
        df (Pandas.DataFrame): Dataframe of interest
        noise (float): Add noise to `confidence` variable.
            Useful to get a nicer swarmplot.
    
    Returns:
        Pandas.DataFrame
    
    """

    info = [
        "floodmaps", 
        "waterdepth", 
        "WTS", 
        "WTScomp", 
        "claims", 
        "ins_rain", 
        "ins_primary", 
        "ins_secondary"
        ]
    infoconf_cols = [f"{i}_conf" for i in info]
    indicator_cols = df.columns[df.columns.str.endswith("indicator")].tolist()

    collist = [indicator_cols, infoconf_cols]
    colnames = ["information", "confidence"]
    patterns = ['|'.join(['friction_', '_indicator']), "_conf"]
    dfs = []

    for cols, colname, pattern in zip(collist, colnames, patterns):

        df_temp = (df[cols + ["uniqueadd_id"]]
            .melt(id_vars=["uniqueadd_id"])
            .rename(columns={"value": colname}))
        df_temp["variable"] = df_temp["variable"].str.replace(pattern, "", regex=True)
        dfs.append(df_temp)
    

    df1, df2 = dfs[0], dfs[1]

    plot_df = df1.merge(df2, on=["uniqueadd_id", "variable"]).dropna()
    plot_df.variable = plot_df.variable.replace({
        "floodmaps": "Flood maps",
        "waterdepth": "Flood maps", 
        "WTS": "Govt. compensation",
        "WTScomp": "Govt. compensation",
        "claims": "Govt. compensation",
        "ins_primary": "Insurance",
        "ins_secondary": "Insurance",
        "ins_rain": "Insurance"
    })
        
    for i in ["information", "confidence"]:
        plot_df[f"{i}_for_plot"] = plot_df[i]

    plot_df_subsets = []
    for q in ["Insurance", "Govt. compensation", "Flood maps"]:
        subset_df = plot_df.query("variable == @q")
        subset_df = scatter_dataframe(subset_df, ["confidence_for_plot", "information_for_plot"], noise)
        plot_df_subsets.append(subset_df)
    
    plot_df = pd.concat(plot_df_subsets)

    plot_df.information = plot_df.information.replace(
        {1: "Incorrect answers", 0: "Correct answers"}
    )
    
    return plot_df


def plot_information_vs_confidence(df, noise=0):
    """Create a Seaborn swarmplot showing confidence level
    for incorrect vs. correct answers.
    
    """
    df = get_df_for_plotting(df, noise)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # create swarmplot
    sns.swarmplot(
        ax=ax,
        data=df, 
        x="confidence_for_plot", 
        y="information", 
        hue="variable", 
        dodge=True, 
        orient="h", 
        size=1,
        linewidth=0,
        palette="deep",
        )

    # add line to separate incorrect vs. correct answers
    ax.hlines(0.5, 1, 10, colors="grey", linewidth=0.5)

    # set axis and axis' labels
    ax.set_ylabel("")
    ax.set_xlabel("Confidence", fontweight="bold", labelpad=10)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
    ax.tick_params(axis='y', which='major', pad=10)
    plt.yticks(
        rotation = 90, ha="center", rotation_mode="anchor"
    )
    sns.despine()

    # adjust legend
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), title="")
    
    # annotate plot with summary stats
    conf_corr = df.query("information == 'Correct answers'").confidence
    conf_inc = df.query("information == 'Incorrect answers'").confidence

    means = [conf.mean().round(2) for conf in [conf_corr, conf_inc]]
    stds = [conf.std().round(2) for conf in [conf_corr, conf_inc]]
    pvalue = stats.ttest_ind(conf_corr.values, conf_inc.values)[1].round(2)

    plt.annotate(
        f"Mean corr.: {means[0]} \nMean incor.: {means[1]} \n(p-value: {pvalue}) \n \nSD corr.: {stds[0]} \nSD incor.: {stds[1]}", 
        xycoords="axes fraction",
        xy=(1.01, 0.4),
        fontsize=8.5,
        bbox=dict(facecolor='none', edgecolor='lightgrey', linewidth=0.5, pad=5.0))

    # set x-axis ticks for 1, 2 ... 10
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    return fig
