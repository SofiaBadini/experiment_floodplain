"""Module to compute descriptive statistics (first step of the analysis)."""

import itertools
import scipy.stats
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind


def make_table_adminlevel_residents(gdf, adminLevels):
    """Compute number of residents in `gdf` dataset for each administrative level in `adminLevels`."""

    residentsDict = {}

    for adminLevel in adminLevels:

        ADMINLEVEL = adminLevel.upper()
        year = 2020 if ADMINLEVEL.startswith("PC") else 2019
        residentsDict[ADMINLEVEL] = {}

        resSeries = gdf[f"TOT_RESIDENTS_{ADMINLEVEL}_{year}"]
        resMean = resSeries.mean()
        resMin = resSeries.min()
        resMax = resSeries.max()
        resSD = resSeries.std()

        for label, stat in zip(
            ["Mean", "Min", "Max", "SD"], [resMean, resMin, resMax, resSD]
        ):

            residentsDict[ADMINLEVEL][label] = "{:,.0f}".format(stat)

    resGDF = pd.DataFrame.from_dict(residentsDict)

    return resGDF


def make_table_unconditional_flood(
    floodareasDF, flood_dict, return_periods, waterdepth_labels, extended=False
):
    """Compute the share of addresses that flood, under all scenarios, at
    different maximum inundation depths.

    Probabilities are unconditional because an address is not removed from less
    likely flood scenario conditionally on flooding under a more likely one.
    For example, an address that floods once in 10 years is also included
    among those that flood once every 100 years. As a result, the same address
    can be included in the share of flooded houses under multiple scenarios.

    Args:
        floodareasDF (pandas.DataFrame): Dataframe from which to compute shares.
            It is supposed to be the dataframe saved as `addressesflooddata.csv`
            in *bld.data*.
        flood_dict (dict): Empty dictionary to populate. Keys need to represent
            the flood scenarios, e.g. "1 in 10 years", "1 in 100 years",
            "1 in 1000 years", "1 in 10000 years".
        return_periods (list or tuple): Can be all or some values in
            (10, 100, 1000, 10000).
        waterdepth_labels (list or tuple): Can be all or some values in
            ("less than 0.5m", "between 0.5 and 1m", "between 1 and 1.5m",
            "between 1.5 and 2m", "between 2 and 5m", "more than 5m").
        extended (bool): If True, considers also houses within 500m from
            flooded areas.

    Returns:
        pandas.DataFrame

    """
    prefix = "CLOSEST_" if extended else ""

    # First loop: for each flood risk scenario determined by return period
    for key, return_period in zip(flood_dict.keys(), return_periods):
        # select addresses which are flooded under given return period
        floodDF = floodareasDF.query(f"{prefix}FLOOD_{return_period} == 1")
        # compute how many they are...
        flooded_addresses = len(floodDF)
        # ... and store the result
        flood_dict[key]["Total"] = "{:,.0f}".format(flooded_addresses)

        # second loop: for each water depth level
        for waterdepth in waterdepth_labels:
            # count how many addresses get flooded to this water level...
            waterdepth_addresses = len(
                floodDF.query(f"{prefix}WATERDEPTH_{return_period} == @waterdepth")
            )
            # ... and store the result
            waterdepth_addresses_formatted = "{:,.0f}".format(waterdepth_addresses)
            # compute again as share of the total addresses under the first loop's flood risk scenario...
            waterdepth_share = (waterdepth_addresses / flooded_addresses) * 100
            # ... and store the result
            waterdepth_share_formatted = "{:,.2f}%".format(waterdepth_share)
            flood_dict[key][
                waterdepth.capitalize()
            ] = f"{waterdepth_addresses_formatted} ({waterdepth_share_formatted})"

    # Finally, create dataframe with results
    waterdepthDF = pd.DataFrame.from_dict(flood_dict)

    return waterdepthDF


def make_table_conditional_flood(
    floodareasDF, return_periods, waterdepth_labels, extended=False
):
    """Compute the share of addresses that flood for the most likely return period
    in `return_periods`, at different maximum inundation depth. Then, for the
    same subsample of addresses, compute again the shares of addresses that flood
    at different inundation depths under all less likely scenarios.

    Args:
        floodareasDF (pandas.DataFrame): Dataframe from which to compute shares.
            It is supposed to be the dataframe saved as `addressesflooddata.csv`
            in *bld.data*, restricted so that all observations have value 1
            in the column `FLOODABLE`.
        return_periods (list or tuple): Can be all or some values in
            (10, 100, 1000, 10000).
        waterdepth_labels (list or tuple): Can be all or some values in
            ("less than 0.5m", "between 0.5 and 1m", "between 1 and 1.5m",
            "between 1.5 and 2m", "between 2 and 5m", "more than 5m").
        extended (bool): If True, considers also houses within 500m from
            flooded areas.

    Returns:
        pandas.DataFrame

        """
    prefix = "CLOSEST_" if extended else ""
    most_likely_return_period = np.min(return_periods)
    keyword = f"1 in {np.min(return_periods)} years"

    # create dictionary
    flood_dict = {}
    for return_period in return_periods:
        flood_dict[
            (
                f"Most likely scenario: 1 in {most_likely_return_period} years",
                f"1 in {return_period} years",
            )
        ] = {}

    # select addresses that flood for the first time under this return period
    floodDF = floodareasDF.query(f"{prefix}FLOOD_MAX == @keyword").copy()
    less_likely_return_periods = [
        x for x in (10, 100, 1000, 10000) if x >= most_likely_return_period
    ]

    waterdepthDF = make_table_unconditional_flood(
        floodDF, flood_dict, less_likely_return_periods, waterdepth_labels, extended
    )

    return waterdepthDF


def make_table_conditional_flood_all_scenarios(
    floodareasDF, waterdepth_labels, extended=False
):
    """Compute the share of addresses that flood for the most likely return period
    in `return_periods`, at different maximum inundation depth. Then, for the
    same subsample of addresses, compute again the shares of addresses that flood
    at different inundation depths under all less likely scenarios. Afterwards,
    repeat the previous two steps for all the return periods in `return_periods`,
    after removing the previous subsample of addresses.

    Probabilities are conditional because, in the third step, an address is removed
    from less likely flood scenarios conditionally on flooding under a more likely one.
    For example, an address that floods once in 10 years is not included
    among those that flood once every 100 years. We then follow the (discrete)
    distribution function of maximum water depth under different scenarios for
    four separate "cohorts" of addresses (those that flood at most once every 10
    years, those that flood at most once every 100 years, those that flood at
    most once every 1,000 years, those that flood at most once every 10,000 years).

    Args:
        floodareasDF (pandas.DataFrame): Dataframe from which to compute shares.
            It is supposed to be the dataframe saved as `addressesflooddata.csv`
            in *bld.data*, restricted so that all observations have value 1
            in the column `FLOODABLE`.
        waterdepth_labels (list or tuple): Can be all or some values in
          ("less than 0.5m", "between 0.5 and 1m", "between 1 and 1.5m",
          "between 1.5 and 2m", "between 2 and 5m", "more than 5m").
        extended (bool): If True, considers also houses within 500m from
            flooded areas.

    Returns:
        pandas.DataFrame

    """
    return_periods = [10, 100, 1000, 10000]
    dfs = [
        make_table_conditional_flood(
            floodareasDF, return_periods[i:], waterdepth_labels, extended
        )
        for i in range(0, len(return_periods))
    ]

    # join dataframes for each scenarios
    df = dfs[0].join([dfs[1], dfs[2], dfs[3]], how="outer")

    return df


def make_table_housingtype(df_names, dfs):
    """Compute number and shares of BAG houses by type, for each dataframe in `dfs`.

    Args:
        dfs (list of pandas.Dataframes): Dataframes.
        df_names (list of str): Name of dataframes in `dfs`.

    Returns:
        pandas.DataFrame

    """

    table_dict = {}

    for df_name, df in zip(df_names, dfs):

        total_houses = len(df)
        housingtype_values = df["HOUSINGTYPE"].value_counts(dropna=False)
        housingtype_shares = housingtype_values / total_houses * 100
        total_share = sum(housingtype_values) / total_houses * 100

        table_dict[(f"{df_name}", "Type of house")] = housingtype_values.to_dict()
        table_dict[(f"{df_name}", "Type of house")]["Total"] = total_houses
        table_dict[(f"{df_name}", "Share")] = housingtype_shares.to_dict()
        table_dict[(f"{df_name}", "Share")]["Total"] = total_share

    table = pd.DataFrame.from_dict(table_dict)

    for df_name in df_names:
        table[(f"{df_name}", "Type of house")] = table[
            (f"{df_name}", "Type of house")
        ].map("{:,.0f}".format)
        table[(f"{df_name}", "Share")] = table[(f"{df_name}", "Share")].map(
            "{:,.2f}%".format
        )

    return table


def make_table_exposure(dfs, df_names):
    """Compute number and share of exposed and not exposed areas for each dataframe in `df`.

    Args:
        dfs (list of pandas.DataFrames or geopandas.GeoDataFrames): Dataframes.
        df_names (list of str): One for each dataframe in `dfs`.

    Returns:
        pandas.DataFrame

    """
    to_compute = ["Count", "Share"]
    table_dict = {i: {} for i in itertools.product(df_names, to_compute)}

    for df_name, df in zip(df_names, dfs):

        vals = list(itertools.product(range(2), repeat=2))
        vals.append((1, 1))
        ops = ["and"] * 4 + ["or"]
        rownames = [
            "Not exposed",
            "Exposed, according to ENW only",
            "Exposed, according to RVO only",
            "Exposed, according to both ENW and RVO",
            "Exposed, according to either ENW or RVO (total exposed)",
        ]
        houses_total = df.shape[0]
        houses_total_check = 0

        for rowname, val, op in zip(rownames, vals, ops):

            houses_count = df.query(
                f"FLOODED_RVO == @val[0] {op} FLOODED_ENW == @val[1]"
            ).shape[0]
            houses_share = houses_count / houses_total * 100

            table_dict[(f"{df_name}", "Count")][rowname] = "{:,.0f}".format(
                houses_count
            )
            table_dict[(f"{df_name}", "Count")]["Total"] = "{:,.0f}".format(
                houses_total
            )
            table_dict[(f"{df_name}", "Share")][rowname] = "{:,.2f}%".format(
                houses_share
            )

            if rowname != "Exposed, according to either ENW or RVO (total exposed)":

                houses_total_check += houses_count

        table_dict[(f"{df_name}", "Share")]["Total"] = "{:,.2f}%".format(
            houses_total_check / houses_total * 100
        )

    rownames.insert(-1, "Total")
    table = pd.DataFrame.from_dict(table_dict).reindex(rownames)

    return table


def make_table_missings(df_names, dfs, col_dict):
    """Compute count and share of missing values in `dfs` for columns specified in `col_dict` dictionary,
    as well as t-statistic and p-values for the mean difference of the two missing values distributions.

    Args:
        df_names (list of str): Name of two dataframes in `dfs`.
        dfs (list of pandas.Dataframes): List of two dataframes.
        col_dict (dict): Dictionary where they keys are the index labels of the pandas.DataFrame
            to be produces by this function, and values are the columns in `dfs` missing values should
            be computed for.

    Returns:
        pandas.DataFrame

    """

    table_dict = {}

    for first_key in col_dict.keys():
        table_dict[first_key] = {}
        nans_values = []

        for df_name, df in zip(df_names, dfs):
            for second_key in col_dict[first_key].keys():

                where_nans = df[second_key].isna()
                nans = df[where_nans].shape[0]
                total = df[second_key].shape[0]
                nans_share = nans / total * 100
                table_dict[first_key].update({df_name: "{:,.2f}%".format(nans_share)})

                nans_distribution = [1 if obs is False else 0 for obs in where_nans]
                nans_values.append(nans_distribution)

        res = ttest_ind(nans_values[0], nans_values[1], equal_var=False)
        table_dict[first_key].update({"t-statistic": res[0]})
        table_dict[first_key].update({"p-value": res[1]})

    table = pd.DataFrame.from_dict(table_dict).replace({np.nan: "-"}).T

    return table


def make_table_balance(df_names, dfs, col_dict):
    """Compute average values in `dfs` for columns specified in `col_dict` dictionary,
    as well as t-statistic and p-values for the mean difference of the two distributions.

    Args:
        df_names (list of str): Name of two dataframes in `dfs`.
        dfs (list of pandas.Dataframes): List of two dataframes.
        col_dict (dict): Dictionary where they keys are the index labels of the pandas.DataFrame
            to be produces by this function, and values are the columns in `dfs` missing values should
            be computed for.

    Returns:
        pandas.DataFrame

    """

    table_dict = {}

    for first_key in col_dict.keys():
        table_dict[first_key] = {}
        values = []

        for df_name, df in zip(df_names, dfs):
            for second_key in col_dict[first_key].keys():

                col_values = df[second_key]
                mean = col_values.mean()

                if second_key == "CONSTRUCTIONYEAR":
                    format_str = "{:.0f}"
                else:
                    format_str = "{:,.2f}"

                table_dict[first_key].update({df_name: format_str.format(mean)})

                values.append(col_values)

        res = ttest_ind(values[0], values[1], nan_policy="omit", equal_var=False)
        table_dict[first_key].update({"t-statistic": res[0]})
        table_dict[first_key].update({"p-value": res[1]})

    table = pd.DataFrame.from_dict(table_dict).replace({np.nan: "-"}).T

    return table


def make_table_target_population(gdf, targetDict):
    """Compute average values in `gdf` for variables in `targetDict`.

    Args:
        gdf (pandas.DataFrame or geopandas.GeoDataFrame): Dataframe of interest.
        targetDict (dict): Dictionary where they keys are the index labels of
            the pandas.DataFrame to be produces by this function, and values are
            the columns in `dfs` average values should be computed for.

    Returns:
        pandas.DataFrame

    """

    tableDict = {}

    for firstKey in targetDict.keys():
        tableDict[firstKey] = {}
        values = []

        for secondKey in targetDict[firstKey].keys():

            colValues = gdf[secondKey]
            mean = colValues.mean()

            if secondKey == "CONSTRUCTIONYEAR":
                formatStr = "{:.0f}"
            else:
                formatStr = "{:,.2f}"

            tableDict[firstKey].update({"Target population": formatStr.format(mean)})

            values.append(colValues)

    table = pd.DataFrame.from_dict(tableDict).replace({np.nan: "-"}).T

    return table


def get_covariates_dataset(covariatesDF, valuesDF, weights=False):
    """Compute (weighted) average values of variables in `covariatesDF` from
    `valuesDF`. Missing values are automatically excluded.

    Args:
        covariatesDF (pandas.DataFrame): Dataframe of covariates. Need to have
            columns named "CATEGORY", "DESCRIPTION", "VARNAME",
            and "WEIGHTS".
        valuesDF (pandas.DataFrame): Dataframes of covariates' values. Need to
            have a column named "VARNAME" containing the variables in `covariatesDF`.
        weights (Bool): Weights to compute weighted mean. If True, will be
            pulled from `valuesDF.WEIGHTS`. Default is False

    Returns:
        pandas.DataFrame

    """

    meanDict = {}

    # for each covariate in the DataFrame of covariates
    for i, row in covariatesDF.iterrows():

        key = (f"{row.CATEGORY}", f"{row.DESCRIPTION}")

        # if VARNAME is in `valuesDF`
        if any(row.VARNAME in x for x in valuesDF.columns.tolist()):

            # select column
            var = valuesDF[row.VARNAME]
            # create a masked array that excludes np.NaNs
            maskedVar = np.ma.MaskedArray(var, mask=np.isnan(var))
            # compute (weighted) mean
            value = (
                var.mean()
                if weights is False
                else np.ma.average(maskedVar, weights=valuesDF.WEIGHTS)
            )
            # update dictionary of results 
            meanDict.update({key: value.round(2)})

    meanDF = pd.DataFrame(meanDict, index=["Share"]).T

    return meanDF


def get_maps_frictions(df, mapinfo, mapdict):
    """Compare correct vs. stated information based on flood maps."""

    mapDF = (pd.DataFrame(df[[f"correct_{mapinfo}", f"stated_{mapinfo}"]]
    .replace(mapdict) 
    .groupby(f"correct_{mapinfo}")
    .value_counts(normalize=True)
    .round(2)
    )
        .reset_index()
        .rename(columns={
            0: "share", 
            f"correct_{mapinfo}": "correct", 
            f"stated_{mapinfo}": "stated"}
            )
        .set_index(["correct", "stated"]))
    
    return mapDF


def _add_dummy_columns(df):
    """Add flood profile dummy columns to Pandas.DataFrame.
    The dummy columns have prefix "waterdepth_max" and "flood_max".
    
    Args:
        df (Pandas.DataFrame): Needs to have the following columns:
            - WATERDEPTH_MAX as indicator (takes values from 1 to 6).
            - FLOOD_MAX as text variables (takes value "1 in 100 years",
                "1 in 1000 years", "1 in 10000 years").
    
    Returns:
        Pandas.DataFrame.
        
    """
    df["WATERDEPTH_MAX"] = df["WATERDEPTH_MAX"].replace(
        {
            1: "less than 0.5m",
            2: "between 0.5 and 1m",
            3: "between 1 and 1.5m",
            4: "between 1.5 and 2m",
            5: "between 2 and 5m",
            6: "more than 5m",
        }
    )
    # create max. water depth indicators
    waterdepth_dummies = pd.get_dummies(df["WATERDEPTH_MAX"], prefix="waterdepth_max")
    waterdepth_dummies.columns = waterdepth_dummies.columns.str.replace(
        " ", "_", regex=False
    ).str.replace(".", "", regex=False)

    # create floood risk indicators
    flood_dummies = pd.get_dummies(df["FLOOD_MAX"], prefix="flood_max")
    flood_dummies.columns = flood_dummies.columns.str.replace(" ", "_", regex=False)
    
    # concatenate dummies DataFrames to original DataFrame
    df = pd.concat([df, waterdepth_dummies, flood_dummies], axis=1)

    return df


def get_spearman_rho(df, varlist):
    """Compute Spearman's rho for `df` columns in `varlist`.
    
    Args:
        df (Pandas.DataFrame): Must contains columns in `varlist`.
        varlist (2-dimensional list): List of paired names 
            of columns the Spearman's rho should be computed for,
            e.g.: [["A", "B"], ["C", "D"]] will return association
            coefficient between A and B, and C and D.
    
    Returns:
        List of floats.

    """

    res = []

    for vars in varlist:
        # drop missing values
        sp_df = df[vars].dropna()
        # compute Spearman's rho
        corr, pval = scipy.stats.spearmanr(
            sp_df[vars[0]], sp_df[vars[1]])
        # saves values (including p-value) to list 
        res.append([vars[0], vars[1], corr.round(2), pval.round(2)])

    return res


def get_summary_stats(df, beliefs, colnames=False):
    """Get summary statistics for `beliefs` columns in `df`. 

    Summary statistics are: minimum value, 25th quantile, mean,
    median, 75th quantile, maximum value, standard deviation (SD)
    and Mean Absolute Deviation (MAD).
    
    """

    colnames = beliefs if colnames is False else colnames

    sdict = {}

    sdict["min"] = df[beliefs].min().tolist()
    sdict["25th quantile"] = df[beliefs].quantile(0.25).tolist()
    sdict["mean"] = df[beliefs].mean().tolist()
    sdict["median"] = df[beliefs].median().tolist()
    sdict["75th quantile"] = df[beliefs].quantile(0.75).tolist()
    sdict["max"] = df[beliefs].max().tolist()

    sdict["SD"] = df[beliefs].std().tolist()
    sdict["MAD"] = (df[beliefs] - df[beliefs].mean()).abs().mean().tolist()

    statsDF = pd.DataFrame.from_dict(sdict, orient="index", columns=colnames)
    statsDF = statsDF.round(2)

    return statsDF


def get_conditional_summary_stats(df, belief, col, vals, colnames):
    """Compute summary statistics of column `belief` in `df`, 
    conditional on `col` taking each value in `vals` list. 

    Summary statistics are: minimum value, 25th quantile, mean,
    median, 75th quantile, maximum value, standard deviation (SD)
    and Mean Absolute Deviation (MAD).

    Args:
        df (Pandas.DataFrame): DataFrame containing column `belief`.
        belief (str): Name of column of which summary statistics should
            be computed.
        col (str): Name of column conditional on which summary statistics 
            for `belief` should be computed.
        vals (list): Values taken by `col`.
        colnames (list of str): Name of columns in DataFrame of results 
            (one name for each value in `vals`). 
    
    Returns:
        Pandas.DataFrame.
        
    """

    statsDFs = []

    for val, colname in zip(vals, colnames):

        condDF = df.query(f"{col} == @val")
        if len(condDF) > 0:
            statsDF = get_summary_stats(condDF, belief, [colname]) 
            statsDFs.append(statsDF)

    condstatsDF = pd.concat(statsDFs, axis=1)

    return condstatsDF


def compute_share_by_columns(df):
    """Replace values of each `df` column with its share 
    with respect to the column total. Add row with total."""
    tot_respondents = df.sum(axis=0).values
    df = df / tot_respondents * 100
    df.loc["tot_respondents"] = tot_respondents
    df = df.round(1)

    return df


def compute_stats_by_treatment(df, columns):
    """Get mean, median, and standard deviation of `columns`
    in Pandas.DataFrame `df`.

    """

    # get mean and rename index
    meanDF = pd.concat([
            df.groupby("treatment")[column].mean().round(1).T
            for column in columns
        ])
    meanDF.index = [f"{i}_mean" for i in meanDF.index]
    # get median and rename index
    medianDF = pd.concat([
            df.groupby("treatment")[column].median().round(1).T
            for column in columns
        ])
    medianDF.index = [f"{i}_median" for i in medianDF.index]
    # get standard deviation and rename index
    stdDF = df = pd.concat([
            df.groupby("treatment")[column].std().round(1).T
            for column in columns
        ])
    stdDF.index = [f"{i}_std" for i in stdDF.index]

    # concatenate to unique dataframe
    statsDF = pd.concat([meanDF, medianDF, stdDF])
    statsDF = statsDF.rename(columns={
        1: "decoy", 2: "maps", 3: "WTS", 4: "insurance"
        })

    return statsDF


def melt_friction_vs_clicks(df, topics):
    """Create long dataframe linking, for each participants,
    information frictions by topic (in list `topics`) to 
    whether the participant clicked on the associated 
    topic box.
    
    """

    dfs = []

    for topic in topics:
        # get subset of id, treatment, plus 
        # topic-specific frictions and clicks 
        subset = (df[[
            "uniqueadd_id", 
            f"friction_topic_{topic}", 
            f"clicks_{topic}_indicator", 
            "treatment"
            ]] 
                .copy().rename(columns={
                    f"friction_topic_{topic}": "friction", 
                    f"clicks_{topic}_indicator": "click"
                    }
                )
            )
        # add topic name
        subset["topic"] = topic

        dfs.append(subset)

    # concatenate (each row is one respondent x one topic)
    df_long = pd.concat(dfs)

    return df_long


def compute_updating_stats(df, belief): 
    """Compute share of `belief` updates, share of updates by direction, 
    and average update size by direction conditional on treatment.
    
    """
    # share of updates by treatment, regardless of direction
    updates_df = df.groupby("treatment")[f"{belief}_update_any"].mean().reset_index()

    # share of updates by direction and treatment
    revise_df = (df.query(f"{belief}_update_any == 1")
        .groupby("treatment")[f"{belief}_revise"]
        .value_counts(normalize=True).reset_index())
    
    _belief = "damages_1000" if belief == "damages" else belief
    
    # average update by treatment and update direction
    average_update_df = (df.query(f"{belief}_update_any == 1")
        .groupby(["treatment", f"{belief}_revise"])[f"{_belief}_update"]
        .mean()
        .reset_index()
        .rename(columns={"risk_update": "average_risk_update"})
        )
    
    # merge everything
    updates_df = updates_df.merge(revise_df, on="treatment").set_index("treatment")
    updates_df = updates_df.merge(average_update_df, on=["treatment", f"{belief}_revise"])
    updates_df = updates_df.set_index("treatment")

    return updates_df
    
 
def compute_share_of_updates(df, belief):
    """Compute share of `belief` update, for expected updates given baseline information
    frictions.

    Args:
        df (Pandas.DataFrame): Dataframe.
        belief (str): Belief.
    
    Returns:
        Pandas.DataFrame.

    """

    df1 = df.groupby("treatment")[f"{belief}_revise_expected"].mean().reset_index()
    df2 = (df.query(f"{belief}_revise_expected == 1")
        .groupby("treatment")[[f"{belief}_should_revise", f"{belief}_revise"]]
        .value_counts(normalize=True)
        .reset_index()
        )
    directions_df = df1.merge(df2, on="treatment")
    directions_df = directions_df.set_index("treatment")
    directions_df

    return directions_df


def compute_mean_and_ttest(df, var, split_by):
    """Split data under column `var` of `df` by the binary
    variable `split_by`, compute the mean of the two groups and 
    test for their differrence (t-test, the variance of the two 
    samples is not assumed to be equal).

    """

    res_df = df.groupby(split_by)[var].mean()
    res_df.index = [f"{res_df.index.name}_{index}" for index in res_df.index]
    res_df = res_df.T

    split_by_vals = df[split_by].value_counts().index.tolist()
    split_by_vals.sort()

    df = df[[split_by, var]].dropna()
    res_df["ttest_pval"] = ttest_ind(
        df[df[split_by] == split_by_vals[0]][var],
        df[df[split_by] == split_by_vals[1]][var],
        equal_var=False
    ).pvalue

    return res_df