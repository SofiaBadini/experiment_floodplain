"""Visualization module for ``task_plot_wtp.py``."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_insurance_two_arms(data):
    """Plot who stays and leaves the flood insurance market, by flood risk 
    category and treament arms ("Neutral text" vs. "Risk profile"). 
    Include average WTP by flood risk category for those who stay.

    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    axs = axs.flatten()
    legends = [True, False]
    labels = ["Neutral text", "Risk profile"]
    for i, (ax, legend, label) in enumerate(zip(axs, legends, labels)):

        t = i + 1
        data_t = data[data["treatment"] == t]
        cat_order = ["1 in 100 years", "1 in 1000 years", "1 in 10000 years"]
        
        sns.histplot(
            ax=ax,
            data=data_t, 
            y="is_wtp_ins_positive", 
            stat="percent", 
            hue="correct_floodmaps", 
            hue_order=cat_order,
            palette="Blues_r",
            discrete=True,
            legend=legend,
            alpha=.75,
            shrink=.6,
            lw=0,
            multiple="stack")
        for i in [0, 2, 4]:
            ax.patches[i].set_hatch('//////')
            ax.patches[i].set_edgecolor('lightgrey')

        ax.set_ylabel(label, labelpad=35, fontweight="bold", rotation=0, fontsize="large")
        ax.set_xlabel("Share of respondents in treatment arm")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["WTP = 0", "WTP > 0"])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xticks(np.arange(0, 80, 5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # reverse category order because bars are plotted from top to bottom!
        cat_order.reverse()
        wtps = (data_t.query("is_wtp_ins_positive == 1").groupby("correct_floodmaps")
            .wtp_insurance_wins975.mean().reindex(cat_order).values)

        for bars, wtp in zip(ax.containers, wtps):
            datavalue = bars.datavalues
            tot = [
                np.sum([bars.datavalues[0] for bars in ax.containers]),
                np.sum([bars.datavalues[1] for bars in ax.containers])
            ]
            labels = [f"{int(x)}%" for x in np.round(datavalue / tot * 100)]
            labels[1] = f"{labels[1]} \nWTP: {np.round(wtp, decimals=2)}"

            ax.bar_label(
                bars, 
                labels=labels,
                label_type='center',
                fontsize='medium',
                )
        if legend == True:
            sns.move_legend(
                ax, "upper center", frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=3, title="Flood risk category")

    axs[0].spines['bottom'].set_visible(False)

    return fig



def plot_wtp_distributions(df):

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["grey", "tab:blue", "tab:orange", "tab:green"]
    sns.kdeplot(df, ax=ax, x="wtp_insurance_wins99", hue="treatment", palette=colors, fill=True, alpha=.25)
    ax.set_xlabel("WTP for insurance", labelpad=10)

    for t, tname, y in zip([2, 3, 4], ["Risk profile", "Govt. compensation", "Insurance"], [.85, .8, .75]):

        statistic, pvalue = scipy.stats.kstest(
            df.query("treatment == 1").wtp_insurance_wins99.dropna(),
            df.query("treatment == @t").wtp_insurance_wins99.dropna()
            )
        ax.annotate(
            "Kolmogorov-Smirnov test", 
            fontweight="bold", 
            xycoords="axes fraction", 
            xy=(0.3, 0.9))
        ax.annotate(
            f"''Neutral text'' vs. ''{tname}'', statistic: {round(statistic, 3)}, pvalue: {round(pvalue, 3)}",
            xycoords="axes fraction",
            xy=(0.3, y)
            )

        tnames = ["Neutral text", "Risk profile", "Govt. compensation", "Insurance"]
        legend_elements = [
            Patch(facecolor=color, edgecolor=color, alpha=.25, label=tname) 
            for color, tname in zip(colors, tnames)
            ]
        ax.legend(
            handles=legend_elements, 
            loc='center', 
            bbox_to_anchor=(0.5, 1.05), 
            frameon=False, 
            ncol=4
            )

    return fig