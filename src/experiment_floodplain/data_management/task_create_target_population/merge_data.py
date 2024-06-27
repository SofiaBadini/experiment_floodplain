"""This script contains functions to merge data from various sources."""

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm


def load_BAG_chunk(BAGpath, BAGdict, chunk):
    """Read 1 million rows of BAG data .csv file, after skipping the number of
    rows indicated by the `chunk` argument. Read the first row as columns' names.

    Args:
        BAGpath (str or Pathlib object): Path to BAG data.
        BAGdict (dict): Dictionary of labels for columns of BAG dataframe.
        chunk (int): Number of rows to skip.

    Returns:
        Pandas.DataFrame
    """
    bagDF = pd.read_csv(
        BAGpath,
        compression="zip",
        sep=";",
        low_memory=False,
        header=0,
        skiprows=list(range(1, chunk)),
        nrows=1_000_000,
    ).rename(columns=BAGdict)
    # Only keep rows whose province is Limburg, if any
    bagDF = bagDF.query("PROVCODE_2022 == 31").copy()

    return bagDF


def assign_flood_risk(key, RKdatadict, bagGDF, depthDict, max_distance=500):
    """Assign each row in `bagGDF` to flooded or non-flooded status according
    to given scenario of Risicokaart flood maps.

    Args:
        key (str): Refers to Risicokaart flood scenario.
        RKdatadict (dict): Dictionary whose keys refer to the Risicokaart scenarios,
            and whose values refer to the path to the Risicokaart flood maps.
        bagGDF (geopandas.Dataframe): BAG dataset. Needs to have `geometry` column.
        depthDict (dict): Dictionary of water depth labels.
    Returns:
        geopandas.Dataframes
    """
    rkGDF = (
        gpd.read_file(RKdatadict[key])
        .to_crs("epsg:28992")
        .rename(columns={"LEGENDA": f"CLOSEST_WATERDEPTH_{key}"})
        .replace(depthDict)
        .drop(columns=["ror"])
    )
    bagGDF = gpd.sjoin_nearest(
        bagGDF,
        rkGDF,
        max_distance=max_distance,
        distance_col=f"DISTANCE_{key}",
        how="left",
    ).drop(columns=["index_right"])
    bagGDF[f"CLOSEST_WATERDEPTH_{key}"] = bagGDF[f"CLOSEST_WATERDEPTH_{key}"].fillna(
        "0m"
    )
    bagGDF[f"CLOSEST_FLOOD_{key}"] = np.where(
        bagGDF[f"CLOSEST_WATERDEPTH_{key}"] != "0m", 1, 0
    )

    bagGDF[f"DISTANCE_{key}"] = bagGDF[f"DISTANCE_{key}"].fillna(max_distance)

    return bagGDF[
        [
            "uniqueadd_id",
            f"CLOSEST_WATERDEPTH_{key}",
            f"CLOSEST_FLOOD_{key}",
            f"DISTANCE_{key}",
        ]
    ]


def remove_redundant_flood_rows(gdf, key, finalGDF):
    """For geodataframe `gdf`, corresponding to a `key` scenario,
    keep the duplicated row with the highest maximum depth.
    """
    gdf = get_waterdepth_indicator(gdf, waterdepths=[key], closest=True)

    # unique index for all rows in dataframe, so that we can distinguish the redundant rows
    gdf["temp_index"] = gdf.reset_index().index
    duplicated_rows = gdf[gdf.uniqueadd_id.duplicated()]

    for index, duplicated_row in duplicated_rows.iterrows():
        gdf = drop_duplicates_with_lower_waterdepth(gdf, duplicated_row, key)

    if len(gdf) != len(finalGDF):
        raise ValueError("Dataframes lengths do not agree, something is wrong.")

    gdf = gdf.drop(columns=["temp_index"])

    return gdf


def drop_duplicates_with_lower_waterdepth(gdf, duplicated_row, key):
    """For geodataframe `gdf`, corresponding to a `key` scenario,
    drop all the rows with the index of `duplicated_row` besides the one with
    the highest value for `NEAR_WATERDEPTH_{key}_INDICATOR`.
    """

    duplicated_row_id = duplicated_row.uniqueadd_id
    duplicated_temp_indexes = gdf.query(
        "uniqueadd_id == @duplicated_row_id"
    ).temp_index.tolist()
    max_water_depth = gdf.query("uniqueadd_id == @duplicated_row_id")[
        f"CLOSEST_WATERDEPTH_{key}_INDICATOR"
    ].max()
    row_to_keep = gdf.query(
        f"uniqueadd_id == @duplicated_row_id and CLOSEST_WATERDEPTH_{key}_INDICATOR == @max_water_depth"
    )

    row_to_keep_temp_index = [row_to_keep.temp_index.values]
    rows_to_keep = [i for i in duplicated_temp_indexes if i in row_to_keep_temp_index]

    if len(rows_to_keep) != 1 or len(row_to_keep_temp_index) != 1:
        raise ValueError(
            "More than one row should be kept, which means there is something wrong with your function."
        )

    rows_to_drop = [
        i for i in duplicated_temp_indexes if i not in row_to_keep_temp_index
    ]

    for row_to_drop in rows_to_drop:
        gdf = gdf.query("temp_index != @row_to_drop").copy()

    return gdf


def merge_flood_datasets_to_bag(scenarios, RKdatadict, bagGDF):
    """Merge 4 datasets assigning BAG addresse to each of the 4
    Risicokaart scenarios to full BAG data.

    Args:
        scenarios (list of pandas.Dataframes): List of dataframes, one for each
            Risicokaart scenario.
        RKdatadict (dict): Dictionary whose keys refer to the Risicokaart scenarios.
        bagGDF (geopandas.Dataframe): BAG dataset.

    Returns:
        geopandas.Dataframe

    """
    resdict = dict(zip(RKdatadict.keys(), scenarios))
    for key in RKdatadict.keys():
        bagGDF = bagGDF.merge(resdict[key], on="uniqueadd_id", how="left")

    return bagGDF


def get_waterdepth_indicator(gdf, waterdepths, var_name):
    """Convert water depth variable named `var_name_waterdepth` in `gdf` to numeric,
    for scenario in `waterdepths`.

    Args:
        gdf (geopandas.GeoDataFrame): Dataframe.
        waterdepths (list): List of some or all values in 10, 100, 1000, 10000.
        var_name (str): Fixed part of variable name in original dataset.

    Returns:
        geopandas.GeoDataFrame

    """
    # numerical variable to indicate maximum water depth for given flood scenario
    for waterdepth in waterdepths:
        gdf[f"{var_name}_{waterdepth}_INDICATOR"] = gdf[
            f"{var_name}_{waterdepth}"
        ].replace(
            {
                "0m": 0,
                "less than 0.5m": 1,
                "between 0.5 and 1m": 2,
                "between 1 and 1.5m": 3,
                "between 1.5 and 2m": 4,
                "between 2 and 5m": 5,
                "more than 5m": 6,
            }
        )

    return gdf


def get_flood_status(gdf):
    """Get flood risk status for each column in `gdf` (takes value "floodable",
    "nearly floodable", "never flooded").
    """
    # address is floodable when distance to flood geometry is 0
    gdf["FLOOD_STATUS"] = np.where(
        (gdf.DISTANCE_10 == 0)
        | (gdf.DISTANCE_100 == 0)
        | (gdf.DISTANCE_1000 == 0)
        | (gdf.DISTANCE_10000 == 0),
        "floodable",
        # address is nearly floodable when distance to flooded geometry is
        # not zero, but lower than 5m (the "CLOSEST_FLOOD_{scenario}" column
        # returns a match)
        np.where(
            (gdf.CLOSEST_FLOOD_10 == 1)
            | (gdf.CLOSEST_FLOOD_100 == 1)
            | (gdf.CLOSEST_FLOOD_1000 == 1)
            | (gdf.CLOSEST_FLOOD_10000 == 1),
            "nearly floodable",
            "never flooded",
        ),
    )

    return gdf


def get_flood_variables(gdf):
    """Get flood risk variables for `gdf`. In particular, this function generates the
    following variables (where "scenario" takes values 10, 100, 1000, and 10000):

        - FLOOD_{scenario} for `scenario` in (10, 100, 1000, 10000): Whether the
          address is within a Risicokaart flooded geometry under given scenario.
        - WATERDEPTH_{scenario} for `scenario` in (10, 100, 1000, 10000): Maximum
          water depth of the flooded geometry the address is within under given
          scenario.
        - WATERDEPTH_{scenario}_INDICATOR for `scenario` in (10, 100, 1000, 10000):
          Maximum water depth of the flooded geometry the address is within under
          given scenario.

    """
    for scenario in [10, 100, 1000, 10000]:
        gdf[f"WATERDEPTH_{scenario}"] = np.where(
            gdf[f"DISTANCE_{scenario}"] == 0,
            gdf[f"CLOSEST_WATERDEPTH_{scenario}"],
            "0m",
        )
        gdf[f"WATERDEPTH_{scenario}_INDICATOR"] = np.where(
            gdf[f"DISTANCE_{scenario}"] == 0,
            gdf[f"CLOSEST_WATERDEPTH_{scenario}_INDICATOR"],
            0,
        )
        gdf[f"FLOOD_{scenario}"] = np.where(
            gdf[f"DISTANCE_{scenario}"] == 0, gdf[f"CLOSEST_FLOOD_{scenario}"], 0
        )

    return gdf


def get_july_floods_indicators(gdf):
    """Get flood exposure variables for `gdf`. In particular, this function
    generates the following variables:

        - FLOODED_ENW_UNPREDICTED_{suffix}: Areas that were flooded in July 2021 according
            to ENW data, but that never flood according to Risicokaart flood maps.
        - FLOODED_RVO_UNPREDICTED_{suffix}: Areas from which flood damage claims were filed
            after the July 2021 floods, but that never flood according to
            Risicokaart flood maps. Only computed if the RVO data are present.

    The suffix "_STRICT" indicates that we consider unpredictable all the
    floods that happened  outside of the geometries in the Risicokaart maps.
    The suffix "_LAX" indicates that we consider unpredictable all the
    floods that happened farther away than 500m from the geometries
    in the Risicokaart maps.

    """
    gdf["FLOODED_ENW_UNPREDICTED_STRICT"] = np.where(
        (gdf.FLOODED_ENW == 1) & (gdf.FLOOD_STATUS != "floodable"), 1, 0
    )

    gdf["FLOODED_ENW_UNPREDICTED_LAX"] = np.where(
        (gdf.FLOODED_ENW == 1) & (gdf.FLOOD_STATUS == "never flooded"), 1, 0
    )

    if "FLOODED_RVO" in gdf.columns:

        gdf["FLOODED_RVO_UNPREDICTED_STRICT"] = np.where(
            (gdf.FLOODED_RVO == 1) & (gdf.FLOOD_STATUS != "floodable"), 1, 0
        )

        gdf["FLOODED_RVO_UNPREDICTED_LAX"] = np.where(
            (gdf.FLOODED_RVO == 1) & (gdf.FLOOD_STATUS == "never flooded"), 1, 0
        )

        gdf["FLOODED"] = np.where(
            (gdf["FLOODED_RVO"] == 1) | (gdf["FLOODED_ENW"] == 1), 1, 0
        )

    return gdf


def get_conditional_waterdepth(df):
    """Compute minimum water depth, conditional on being flooded, for `df`.
    Requires `df` to have column FLOOD_STATUS, WATERDEPTH_10_INDICATOR,
    WATERDEPTH_100_INDICATOR, WATERDEPTH_1000_INDICATOR, WATERDEPTH_10000_INDICATOR,
    and WATERDEPTH_MAX.
    """

    # select only addresses that flood in some scenario and save unique ids to list
    flooded_addresses_id = df.query(
        "FLOOD_STATUS == 'floodable'"
    ).uniqueadd_id.values.tolist()

    # keep only addresses that flood in some scenario
    col_to_keep = [
        f"WATERDEPTH_{scenario}_INDICATOR" for scenario in (10, 100, 1000, 10000)
    ]
    flooded_addresses_waterdepth = (
        df.query("FLOOD_STATUS == 'floodable'")[col_to_keep].copy().values
    )

    # for each address, keep only waterdepth in scenarios where the address floods
    conditional_waterdepth = [
        waterdepth[np.where(waterdepth != 0)].tolist()
        for waterdepth in flooded_addresses_waterdepth
    ]

    # compute maximum conditional waterdepth
    max_conditional_waterdepth = [
        np.max(waterdepth) for waterdepth in flooded_addresses_waterdepth
    ]

    # compute minimum conditional waterdepth
    min_conditional_waterdepth = [
        np.min(waterdepth) for waterdepth in conditional_waterdepth
    ]

    # create dataframe of results
    waterstatsDF = pd.DataFrame(
        {
            "uniqueadd_id": flooded_addresses_id,
            "WATERDEPTH_MIN": min_conditional_waterdepth,
            "WATERDEPTH_MAX": max_conditional_waterdepth,
        }
    )

    # merge results back to main dataframe
    df = df.merge(waterstatsDF, on="uniqueadd_id", how="left")

    # Addresses that never flood or nearly flood will have nans for this column.
    # Replace with 0
    df["WATERDEPTH_MIN"] = df["WATERDEPTH_MIN"].fillna(0)
    df["WATERDEPTH_MAX"] = df["WATERDEPTH_MAX"].fillna(0)

    return df


def get_most_likely_scenario(df, extended=False):
    """Get most likely scenario for which each addres in `df` floods.
    Setting `extended` to True considers also houses within 500m from
    flooded areas.

    """
    prefix = "CLOSEST_" if extended else ""
    query_keywords = (
        "FLOOD_STATUS == 'floodable' or FLOOD_STATUS == 'nearly floodable'"
        if extended
        else "FLOOD_STATUS == 'floodable'"
    )

    # select only addresses that flood in some scenario and save unique ids to list
    flooded_addresses_id = df.query(query_keywords).uniqueadd_id.values.tolist()

    # keep only addresses that flood in some scenario
    col_to_keep = [f"{prefix}FLOOD_{scenario}" for scenario in (10, 100, 1000, 10000)]
    flooded_addresses_scenarios = df.query(query_keywords)[col_to_keep].copy()

    flood_max = np.where(
        flooded_addresses_scenarios[f"{prefix}FLOOD_10"] == 1,
        "1 in 10 years",
        np.where(
            flooded_addresses_scenarios[f"{prefix}FLOOD_100"] == 1,
            "1 in 100 years",
            np.where(
                flooded_addresses_scenarios[f"{prefix}FLOOD_1000"] == 1,
                "1 in 1000 years",
                "1 in 10000 years",
            ),
        ),
    )

    # create dataframe of results
    floodmaxDF = pd.DataFrame(
        {"uniqueadd_id": flooded_addresses_id, f"{prefix}FLOOD_MAX": flood_max}
    )

    # merge results back to main dataframe
    df = df.merge(floodmaxDF, on="uniqueadd_id", how="left")

    # Addresses that never flood or nearly flood will have nans for FLOOD_MAX column.
    # Replace with 0
    df[f"{prefix}FLOOD_MAX"] = df[f"{prefix}FLOOD_MAX"].fillna(0)

    # equivalent columns with dummies
    dummiesDF = (
        pd.get_dummies(df[f"{prefix}FLOOD_MAX"])
        .rename(
            columns={
                "1 in 10 years": f"{prefix}FLOOD_MAX_10",
                "1 in 100 years": f"{prefix}FLOOD_MAX_100",
                "1 in 1000 years": f"{prefix}FLOOD_MAX_1000",
                "1 in 10000 years": f"{prefix}FLOOD_MAX_10000",
            }
        )
        .drop(columns=[0])
    )

    df = pd.concat([df, dummiesDF], axis=1)

    return df


def assign_address_to_admin_region(addressesGDF, admin_path, admin_level):
    """Assign each address in `addGDF` to specified administrative region.

    I do a spatial merge (i.e. I assign each address to the
    administrative region that contains the shapely.Point associated with
    such address, as recovered from the `lat` and `lon` columns) instead of
    merging on administrative level's code.

    The reason is that I am forced to use data from different years, and
    any administrative level of a certain address (but especially the low-level
    ones) can change over time. For example, postcodes can be reassigned to a
    different buurt or wijk.

    The function returns a dataframe which contains a column equal to the index
    of the merged administrative level dataframe.

    Args:
        addressesGDF (geopandas.DataFrame): Dataframe of addresses. Needs to
             have an active `geometry` column to perform the spatial merge.
        admin_path (str or Pathlib object): Path to geopandas.DataFrame
            of administrative region. Needs to have an active `geometry` column
            to perform the spatial merge.
        admin_level (str): Level of administrative region.
            Can be "pc6", "pc5", "buurt", "wijk", or "gemeente".
        path (str or Pathlib object): Path to save the final dataframe to.

    Returns:
        geopandas.DataFrame

    """
    # load administrative dataset
    adminGDF = gpd.read_file(admin_path).set_geometry("geometry").to_crs(epsg=28992)

    # spatial join with addresses dataset
    mainGDF = gpd.sjoin(
        addressesGDF, adminGDF, predicate="within", how="left"
    ).drop_duplicates()

    if len(mainGDF) != len(addressesGDF):
        raise ValueError("Something went wrong with the merging procedure!")

    if len(mainGDF[mainGDF.index_right.isna()]) != 0:
        raise ValueError(
            "Some addresses cannot be assigned to PC5, you should look into that"
        )

    # include column correponding to the index of the administrative level's dataframe
    mainGDF = mainGDF.rename(columns={"index_right": f"index_{admin_level}"})

    return mainGDF


def assign_admin_geometries_to_addresses(df, admin_path, admin_level):
    """Merge each row of `df` to the appropriate administrative region
    boundary in the `admin_path` geopandas.GeoDataFrame.

    Args:
        df (pandas.DataFrame): Dataframe of BAG addresses, with `index_{admin_level}`
            columns (needed to merge on) and either `LAT` and `LON` columns or
            `geometry` column. It is supposed to be the dataframe saved as `population.csv`
            in *bld.data*.
        admin_path (str or Pathlib object): Path to administrative dataset to load.
             It is supposed to be one of the datasets in *bld.data.CSB*.
        admin_level (str): Administrative level of dataset, for example
            "pc5", "pc6", "buurt", "wijk".

    returns:
        geopandas.GeoDataFrame

    """
    if "geometry_addresses" not in df.columns:
        # create shapely.Points from addresses' latitude and longitude
        df = (
            gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON, df.LAT))
            .rename(columns={"geometry": "geometry_addresses"})
            .set_geometry("geometry_addresses")
            .set_crs(epsg=4326)
            .to_crs(epsg=28992)
        )

    # load dataset of administrative regions
    adminGDF = gpd.read_file(admin_path).set_geometry("geometry").to_crs(epsg=28992)

    # create index column to merge on
    adminGDF[f"index_{admin_level}"] = adminGDF.index

    # merge BAG data to geometries of relevant administrative regions
    gdf = df.merge(
        adminGDF[[f"index_{admin_level}", "geometry"]],
        on=f"index_{admin_level}",
        how="left",
    )

    # rename geometry column
    gdf = gdf.rename(columns={"geometry": f"geometry_{admin_level}"})

    return gdf


def add_floodmaps_indicators(df, waterdepth_column, floodrisk_column):
    """Derive indicators of flood risk and maximum water depth from
    existing columns named `waterdepth_column` and `floodrisk_column` 
    in Pandas.DataFrame `df`.
    
    """
    # convert waterdepth to strings if this is not the case already
    df[waterdepth_column] = df[waterdepth_column].replace(
    {
            0: "0 cm",
            1: "less than 0.5m",
            2: "between 0.5 and 1m",
            3: "between 1 and 1.5m",
            4: "between 1.5 and 2m",
            5: "between 2 and 5m",
            6: "more than 5m" 
        }
    )

    # add water depth indicators
    waterdepth_dummies = pd.get_dummies(
        df[waterdepth_column], 
        prefix="waterdepth_max"
        )
    waterdepth_dummies.columns = (waterdepth_dummies.columns
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "", regex=False))
    
    # add floood indicators
    flood_dummies = pd.get_dummies(df[floodrisk_column], prefix="flood_max")
    flood_dummies.columns = (flood_dummies.columns
        .str.replace(" ", "_", regex=False))
    
    # create derived waterdepth measure for randomization tests
    waterdepth_dummies["waterdepth_max_over_2m"] = np.where(
        (waterdepth_dummies["waterdepth_max_between_2_and_5m"] == 1) |
        (waterdepth_dummies["waterdepth_max_more_than_5m"] == 1), 1, 0)

    # concatenate into single dataframe
    df = pd.concat([df, waterdepth_dummies, flood_dummies], axis=1)

    return df