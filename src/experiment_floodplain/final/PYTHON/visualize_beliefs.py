"""Visualization module for ``task_plot_beliefs.py``."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


def add_variables_for_plotting(df, edges, labels_damages):
    """Add variables for plotting figures."""

    # minor formatting and creation of plot-specific variables
    df["risk_update"] = df["risk_RE"] - df["risk"]
    df["risk_conf_update"] = df["risk_conf_RE"] - df["risk_conf"]
    df["comptot_update"] = df["comptot_RE"] - df["comptot"]
    df["comptot_conf_update"] = df["comptot_conf_RE"] - df["comptot_conf"]
    df["worry_update"] = df["worry_RE_numeric"] - df["worry_numeric"]

    df["coarse_friction_floodmaps"] = (df["friction_floodmaps"].replace(
        {-3: -1, -2: -1, 2: 1, 3: 1})
        )
    df["coarse_friction_waterdepth"] = (df["friction_waterdepth"].replace(
        {-5: -1, -4: -1, -3: -1, -2: -1, 2: 1, 3: 1, 4: 1, 5: 1})
        )
    
    df["risk_for_plot"] = df["risk"].round(0)

    df['damages_for_plot'] = pd.cut(
        df["damages_wins975_1000"], 
        bins=edges, 
        labels=labels_damages, 
        right=False, 
        ordered=True
        )

    df["risk_update_abs"] = np.abs(df["risk_update"])
    df["risk_update_for_plot"] = np.where(
        (df["risk_update_abs"] > -1) &
        (df["risk_update_abs"] < 1) &
        (df["risk_update_abs"] != 0),
        1, df["risk_update_abs"]
    )

    df["risk_update_for_plot"] = df["risk_update_for_plot"].apply(lambda x: _custom_round(x, base=5))
    df["risk_update_for_plot"] = np.where(
        df["risk_revise_expected"] == 0, -df["risk_update_for_plot"], np.where(
            df["risk_revise_expected"] == 1, df["risk_update_for_plot"], np.nan
        ))
    df["risk_update_for_plot"] = df["risk_update_for_plot"].replace(-0, 0)
    df["risk_update_for_plot_cat"] = np.where(df["risk_update_for_plot"] < 0, "wrong", np.where(
        df["risk_update_for_plot"] > 0, 'right', df["risk_update_for_plot"]
    ))
    df["risk_update_for_plot"] = df["risk_update_for_plot"].astype("category")

    return df


def _custom_round(x, base=5): 
    x = base if x < base and x > 0 else x
    x = int(base * round(float(x)/base)) if x == x else x
    return x


def make_jointplot(df, x, y, title, axis_spaced_by_5=False):
    """Make jointplot of varables `x` and `y`
    in Pandas.DataFrame `df`, with `title`.
    """
    fig = sns.jointplot(
        df,
        x=x,
        y=y,
        alpha=0.5,
        marginal_kws=dict(bins=100)
    )

    fig.ax_marg_x.axvspan(
        df[x].quantile(0.25), df[x].quantile(0.75), color='red', alpha=0.1)

    fig.ax_marg_y.axhspan(
        df[y].quantile(0.25), df[y].quantile(0.75), color='red', alpha=0.1)

    fig.refline(x=df[x].mean(), y=df[y].mean(), color='red', ls="solid")
    fig.refline(x=df[x].median(), y=df[y].median(), color='red', ls='--')

    if axis_spaced_by_5:
        plt.xticks(np.arange(0, 105, 5))
        plt.yticks(np.arange(0, 105, 5))

    plt.xlabel("Prior belief", labelpad=10)
    plt.ylabel("Posterior belief", labelpad=10)
    plt.suptitle(title, fontweight="bold", y=1.01)
    plt.axline([0, 0], [1, 1], color="black", lw=1)
    sns.despine()

    return fig


def plot_all_histograms_belief_vs_confidence(
    dict_keys, df, xs, ys, list_of_bins, xlabels, list_of_xticks, labels_dicts, add_missings, type):
    """Plot multiple histograms of beliefs vs. average confidence in beliefs, and add them to a dictionary.
    
    Args:
        dict_keys (list): Keys of dictionary of results, one for each figure.
        df (Pandas.DataFrame): Dataframe with columns to plot.
        xs (list of str): Names of columns to be plotted on the x-axis (values of reported beliefs).
        ys (list of str): Names of columns to be plotted on the y-axis (values of reported confidence in beliefs).
        list_of_bins (list of lists): List of lists of number of bins for histograms.
        xlabels (list of str): List of names for x-axis labels. 
        list_of_xticks (list of lists): List of lists of x-axis coordinates for ticks.
        labels_dicts (list of lists): Whether to rename x-axis ticks.
        add_missings (list of bool): List of whether to include columns with missing values in the final histograms. 
        type (list of str): List of type of plot for average confidence, either "barplot" or "pointplot".

    Returns:
        dictionary.
    
    """

    plots_dict = {}

    for key, x, y, bins, xlabel, xticks, labels_dict, missing in zip(
        dict_keys, xs, ys, list_of_bins, xlabels, list_of_xticks, labels_dicts, add_missings
    ):

        plots_dict[key] = plot_histogram_belief_vs_confidence(
            df=df, 
            x=x, 
            y=y, 
            bins=bins, 
            xlabel=xlabel, 
            xticks=xticks, 
            labels_dict=labels_dict,
            add_missing_values=missing, 
            type=type
        )
    
    return plots_dict


def plot_histogram_belief_vs_confidence(
    df, x, y, xlabel, xticks, bins, labels_dict=False, add_missing_values=False, rotation=0, type="barplot"):
    """"Plot histogram of beliefs vs. average confidence in beliefs.
    
    Args:
        df (Pandas.DataFrame): Dataframe with columns to plot.
        x (str): Name of column to be plotted on the x-axis (values of reported beliefs).
        y (str): Name of columns to be plotted on the y-axis (values of reported confidence in beliefs).
        xlabel (str): Name for x-axis label. 
        xticks (list of int): List of x-axis coordinates for ticks.
        bins (list of int): Number of histogram bins.
        labels_dict (dict): Optional, dictionary of x-axis ticks and x-axis ticks' labels.
        add_missing_values (bool): Whether to include columns with missing values in the final histograms, default is False. 
        rotation (int): Rotation of x-axis ticks, default is 0.
        type (str): Type of plot for average confidence, either "barplot" (default) or "pointplot".

        Returns:
            matplotlib.Figure.
    
    """
    
    df = df[[x, y]].dropna()
    conf_df = _add_missing_values_to_df(df, x, y, xticks) if add_missing_values else df

    fig, axs = plt.subplots(2, 1, figsize=(35, 10), sharex=True)
    axs = axs.flatten()

    # percentage of respondents
    sns.histplot(
        ax=axs[0], 
        data=df, 
        x=x,
        stat="percent", 
        linewidth=0,
        bins=bins,
        discrete=True,
        shrink=.8,
        alpha=.75
        )
    axs[0].set_ylabel("Percentage\nof respondents", labelpad=20, fontsize=32)

    # average confidence
    if type == "barplot":
        sns.barplot(
            ax=axs[1], 
            data=conf_df, 
            x=x,
            y=y,
            errorbar=("ci", 95),
            linewidth=0.5,
            color="lightgrey",
            alpha=.75,
            err_kws={'linewidth': 2, 'color': 'grey'},
            saturation=1
            )
    elif type == "pointplot":
        sns.pointplot(
            ax=axs[1],
            data=conf_df, 
            x=x,
            y=y,
            errorbar=("ci", 95), 
            linestyles='',
            errwidth=2,
            markers="d",
            color="lightgrey")
        
    for yval in [4, 6, 8]:
        axs[1].axhline(y=yval, color='grey', lw=0.5, linestyle='dotted')
    
    axs[1].set_ylabel("Average confidence\nin belief", fontsize=32, labelpad=10)
    axs[1].set_xlabel(xlabel, labelpad=10, fontsize=32, fontweight="bold")
     
    axs[1].set_yticks(np.arange(-1, 11, 1))
    ymin, ymax = axs[1].get_ylim()
    axs[1].set_ylim(0, ymax)
    axs[1].set_ylim(axs[1].get_ylim()[::-1])

    axs[1].set_xticks(xticks)
    if labels_dict:
        xticklabels = [labels_dict[i] for i in xticks]  
        axs[1].set_xticks(axs[1].get_xticks())
        axs[1].set_xticklabels(xticklabels, rotation=rotation)
    else: 
        axs[1].set_xticks(axs[1].get_xticks())
        axs[1].set_xticklabels(xticks)

    axs[1].xaxis.set_tick_params(labelsize=25)
    axs[0].yaxis.set_tick_params(labelsize=25)
    axs[1].yaxis.set_tick_params(labelsize=25)

    sns.despine(left=True)
    plt.subplots_adjust(hspace=0)

    return fig


def _add_missing_values_to_df(df, x, y, xticks):
    """Add -1 as default value of `y` for those values of `x`
    missing from `xticks`. Needed to include missing values
    when plotting histograms.    
    
    Args:
        df (Pandas.DataFrame): Dataframe with columns to plot.
        x (str): Name of column to be plotted on the x-axis (values of reported beliefs).
        y (str): Name of columns to be plotted on the y-axis (values of reported confidence in beliefs).
        xlabel (str): Name for x-axis label. 
        xticks (list of int): List of x-axis coordinates for ticks.

    Returns:
        pandas.DataFrame    
        
    """

    missing_x_vals = [i for i in xticks if i not in df[x].values]
    missing_y_vals = [-1]*len(missing_x_vals)
    missing_dict = dict(zip(missing_x_vals, missing_y_vals))
    missing_vals_df = (pd.DataFrame
        .from_dict(missing_dict, orient="index")
        .reset_index()
        .rename(columns={0:y, "index": x})
    )

    df = pd.concat([df, missing_vals_df]).reset_index().drop(columns="index")

    return df


def get_labels_dictionary(all_ticks, ticks_to_show):
    """Create dictionary of labels for plot."""
    labels = [t if t in ticks_to_show else "" for t in all_ticks]
    labels_dict = dict(zip(all_ticks, labels))

    return labels_dict


def histogram_scatterplot_belief_distributions(df, x, y, xlabel):
    """Plot histogram and scatterplot of `x` vs. `y` from Pandas.DataFrame `df`."""

    data = pd.melt(df[[x, y]]).replace({x: "Prior belief", y: "Posterior belief"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), width_ratios=[3, 1])

    sns.histplot(
        ax=ax1, 
        data=data, 
        x="value", 
        hue="variable", 
        stat="percent", 
        multiple="dodge", 
        bins=100, 
        palette=["tab:blue", "tab:orange"])
    sns.move_legend(
        ax1, 
        title="", 
        ncol=2, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.15), 
        frameon=False,
        fontsize=16
        )
    ax1.set_xlabel(xlabel, fontweight="bold", fontsize=20)
    ax1.set_ylabel("Share", fontweight="bold", fontsize=20)
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=16)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=16)
    for var, color in zip([x, y], ["tab:blue", "tab:orange"]):
        ax1.axvline(x=df[var].mean(), color=color, ls="solid", lw=.75)
        ax1.axvline(x=df[var].median(), color=color, ls="--", lw=.75)
        ax1.axvspan(df[var].quantile(.25), df[var].quantile(.75), color=color, alpha=.2)
    sns.scatterplot(ax=ax2, data=df, x=x, y=y, alpha=.5, color="black")
    ax2.set_xlabel("Prior belief", fontweight="bold", fontsize=20)
    ax2.set_ylabel("Posterior belief", fontweight="bold", fontsize=20)
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=16)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=16)
    ax2.axline([0, 0], [1, 1], color="black", lw=0.5)

    sns.despine()
    plt.subplots_adjust(wspace=0.1)
    
    return fig


def _format_axis_for_directions_plot(ax, xlabel, ylabel):
    """Format x- and y-axis for plot depicting belief updates by direction 
    (implied by baseline information quality).
    
    """

    xticks = ax.get_xticks()
    xtick_middle = [t.get_position() for t in ax.get_xticklabels() if t.get_text() == "0.0"][0][0]
    xtick_min, xtick_max = np.min(xticks), np.max(xticks)
    ax.set_xticks([xtick_min, xtick_middle, xtick_max])
    ax.set_xticklabels([100, 0, 100])
    ax.set_ylim(0, 75)
    ax.annotate(
        '', xy=(xtick_min+1, -7), xytext=(xtick_middle-1, -7), # draws an arrow from one set of coordinates to the other
        arrowprops=dict(arrowstyle='simple',facecolor='black'), # sets style of arrow and colour
        annotation_clip=False)  
    ax.annotate(
        'Unexpected direction', xy=(0, 0), xytext=(xtick_middle-10, -22), # adds another annotation for the text that you want
        fontsize="small",
        annotation_clip=False)
    ax.annotate(
        '', xy=(xtick_max-1, -7), xytext=(xtick_middle+1, -7), # draws an arrow from one set of coordinates to the other
        arrowprops=dict(arrowstyle='simple',facecolor='black'), # sets style of arrow and colour
        annotation_clip=False)  
    ax.annotate(
        'Expected direction', xy=(0, 0), xytext=(xtick_max-10, -22), # adds another annotation for the text that you want
        fontsize="small",
        annotation_clip=False)

    return ax


def plot_updates(df, query_strings, titles, xlabel, ylabel, figsize):
    """Plot belief updates by direction implied by baseline information quality.
    
    Args:
        df: Dataset containing variables of interest.
        query_strings: Strings to select variables of interest.
        titles: Titles of sub-figures.
        xlabel: x-axis label.
        ylabel: y-axis label.
        figsize: Figure size.
    
    Returns:
        matplotlib.Figure.
    
    
    """

    n_subplots = len(titles) 

    fig, axs = plt.subplots(n_subplots, 1, figsize=figsize)
    axs = axs.flatten()

    for ax, query_string, title in zip(axs, query_strings, titles):
        color = "tab:blue" if "treatment == 1" in query_string else "tab:orange"
        sns.countplot(
            df.query(query_string), 
            ax=ax,
            x="risk_update_for_plot", 
            color=color,
            stat="percent"
            )
        nobs = len(df.query(query_string)["risk_update_for_plot"].dropna())
        title = title + f" (N={nobs})"
        ax.set_title(title, fontstyle="italic", y=1.05)

        for type in ("right", "wrong"):

            share, mean = _get_updates(df, query_string, type=type)
            type_ = "Expected" if type == "right" else "Unexpected"

            share_patch = mpatches.Patch(color='None', lw=0, label=f"{type_} updates: {round(share, 2)}%")
            mean_patch = mpatches.Patch(color='None', lw=0, label=f"Average update size: {round(mean, 2)}")

            xpos = .7 if type == 'right' else .2
            legend = ax.legend(
                handles=[share_patch, mean_patch], 
                frameon=False, 
                bbox_to_anchor=(xpos, .65), 
                title="",
                loc="center",
                fontsize="small")
            if type == "right":
                ax.add_artist(legend)

    for ax in axs:    
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax = _format_axis_for_directions_plot(ax, xlabel, ylabel)
        ax.set_yticks([0, 25, 50, 75])
        ax.spines['bottom'].set_linewidth(.5)
        xmin, xmax = ax.get_xlim()
        ax.grid(axis='y', linewidth=.5, color="grey", linestyle="dotted")
        ax.set_axisbelow(True)

    axs[n_subplots - 1].set_ylabel(ylabel, labelpad=7.5, fontweight="bold")
    axs[n_subplots - 1].set_xlabel(xlabel, labelpad=7.5, fontweight="bold")
    
    sns.despine()
    fig.subplots_adjust(wspace=0.05, hspace=.75)

    return fig


def _get_updates(df, query_string, type):
    updates_share = (df
        .query(query_string)["risk_update_for_plot_cat"]
        .dropna()
        .value_counts(normalize=True)[type]*100
    )
    updates_mean = (df
        .query(f"{query_string} and risk_update_for_plot_cat == @type and risk_update_abs != 0")
        .risk_update_abs
        .mean()
    )

    return (updates_share, updates_mean)


def plot_lower_triangular_heatmap(df_corr, suptitle, n_obs, cmap):
    """Plot lower triangular heatmap depicting correlation between
    answers to multiple choice questions (on measures against flood 
    or sources consulted about flood risk).
    
    Args:
        df (Pandas.DataFrame): Dataframe of two-ways correlations.
        suptitle (Str): Plot main title.
        n_obs (int): Number of observations. Will be written in the
            plot sub-title.
        cmap (palette): Seaborn palette.

    Returns:
        Matplotlib.Figure
    
    """
    # get the upper triangle of the co-relation matrix
    matrix = np.triu(df_corr)
    # get maximum and minimum values, excluding 1
    vmax = df_corr.replace(1, -99).max().max()
    vmin = df_corr.min().min()

    # plot heatmap using the upper triangle matrix as mask
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(df_corr, ax=ax, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, mask=matrix)
    plt.suptitle(
        suptitle,
        fontweight="bold",
        y=0.965,
    )
    ax.set_title(f"Observations: {n_obs}")
    plt.xticks(rotation=45, ha="right")

    return fig