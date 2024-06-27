"""This script assigns each address in the clean BAG dataset in *bld/data/BAG*
to (1) flooded areas in July 2021 according to ENW data, (2) 6-digits postcodes 
from which claims for house-related and house content-related damages were filed 
to the RVO after the floods in July 2021, and (3) flood probability and maximum
water depth according to the Risicokaart flood maps. The final dataset is saved as a .csv file 
named ``full_sample.csv`` in *bld/data*.

The RVO data are necessary to create the indicator variable ``FLOODED``, which in turn
is necessary for the sampling of survey recipients. The RVO data are not available 
to the public, but will be available to researchers as a .csv file named ``rvo_pc6.csv`` 
once the paper is published. See also the Introduction of this documentation.

.. note:: 

   This project automatically uses the RVO data if the .csv file is placed under *src/experiment_floodplain/data*.
   Otherwise, the values of the variable ``FLOODED`` are generated at random. 

"""
import pytask
import pandas as pd
import geopandas as gpd
import json
import warnings

from pathlib import Path
from joblib import Parallel, delayed

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.data_management.task_create_target_population.merge_data import (
    assign_flood_risk,
    merge_flood_datasets_to_bag,
    get_waterdepth_indicator,
    get_flood_status,
    get_flood_variables,
    get_conditional_waterdepth,
    get_july_floods_indicators,
    get_most_likely_scenario,
    remove_redundant_flood_rows,
    add_floodmaps_indicators
)

RKpath = BLD / "replication_data" / "RISICOKAART" / "floodmaps2019"
dictpath = SRC / "data_management" / "task_create_target_population" / "json"
depends_on = {
    "depthlevelLabels": dictpath / "depthlevelLabels.json",
    "ENWdata": BLD / "data" / "ENW" / "ENW.GPKG",
    "BAGdata": BLD / "data" / "BAG" / "BAG.csv",
    "10": RKpath / "10depth" / "10depth.shp",
    "100": RKpath / "100depth" / "100depth.shp",
    "1000": RKpath / "1000depth" / "1000depth.shp",
    "10000": RKpath / "10000depth" / "10000depth.shp",
    }

produces = BLD / "data" / "full_sample.csv"

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_assign_flood_exposure(depends_on, produces):
    # load dictionary to rename water depth levels in risicokaart data
    depthDict = json.load(open(depends_on["depthlevelLabels"]))

    # load BAG data
    bagDF = pd.read_csv(depends_on["BAGdata"], low_memory=False, index_col=0)

    ## ONLY IF RVO DATA IS PRESENT!
    path_to_rvo_data = Path(SRC / "data" / "rvo_pc6.csv")
    if path_to_rvo_data.is_file():
        
        # load RVO data
        rvoDF = pd.read_csv(path_to_rvo_data, sep=";").rename(columns={"PC_6": "PC6"})
        rvoDF["FLOODED_RVO"] = 1
        
        # merge RVO damage data to BAG data, on PC6 postcode
        bagDF = bagDF.merge(
            rvoDF[["PC6", "FLOODED_RVO"]], how="left", left_on="PC6_2022", right_on="PC6",
        )
        bagDF["FLOODED_RVO"] = bagDF["FLOODED_RVO"].fillna(0)

    # Convert latitude and longitude to geometry column
    bagGDF = gpd.GeoDataFrame(
        bagDF, geometry=gpd.points_from_xy(bagDF.LON, bagDF.LAT)
    ).set_crs(epsg=4326)

    # change crs to ENW data
    bagGDF = bagGDF.to_crs(epsg=28992)

    # load ENW data and only select flood exposure
    enwGDF = gpd.read_file(depends_on["ENWdata"]).to_crs(epsg=28992)
    floodedGDF = enwGDF.query("FEATURE.str.contains('flood')").copy()
    floodedGDF = floodedGDF.rename(
        columns={
            "COMMENT": "FLOODED_ENW_COMMENT",
            "FEATURE": "FLOODED_ENW_RIVER",
            "STATUS": "FLOODED_ENW",
        }
    )

    # assign flooded area in July 2021, according to ENW data
    # apparently there's some overlapping geometry in the ENW dataset, so I drop
    # duplicated addresses in the merge geodataframe
    bagGDF = (
        gpd.sjoin(bagGDF, floodedGDF, predicate="within", how="left")
        .drop(columns=["index_right"])
        .drop_duplicates()
    )

    bagGDF[["FLOODED_ENW_COMMENT", "FLOODED_ENW_RIVER"]] = bagGDF[
        ["FLOODED_ENW_COMMENT", "FLOODED_ENW_RIVER"]
    ].fillna("None")
    bagGDF["FLOODED_ENW"] = bagGDF["FLOODED_ENW"].fillna(0)

    # create small BAG to merge (only unique ID and geometries)
    bagGDF["uniqueadd_id"] = bagGDF.index
    bagtomergeGDF = bagGDF[["uniqueadd_id", "geometry"]]

    RKdatadict = {
        key: depends_on[key] 
        for key in 
        ["10", "100", "1000", "10000"]
        }

    # assign ex ante flood exposure to small BAG
    floodexpGDFs = Parallel(n_jobs=-2, verbose=20)(
        delayed(assign_flood_risk)(key, RKdatadict, bagtomergeGDF, depthDict)
        for key in RKdatadict.keys()
    )

    # take care of potential duplicates and check if nans are 0 (as it should)
    for i, (gdf, key) in enumerate(zip(floodexpGDFs,  RKdatadict.keys())):
        if len(gdf) > len(bagGDF):

            warnings.warn(
                "There may be duplicates (for example, because some addresses are \
                equidistant to different geometries in the Risicokaart data)"
            )
            # NOTE: this function calls `get_waterdepth_indicator`,
            # so we don't have to compute the indicators later
            floodexpGDFs[i] = remove_redundant_flood_rows(gdf, key, bagGDF)

            # check that the issue is solved
            if len(floodexpGDFs[i]) != len(bagGDF):
                raise ValueError(
                    "Lenghts of dataframes do not agree, something is wrong!"
                )

        else:

            floodexpGDFs[i] = get_waterdepth_indicator(
                gdf, waterdepths=[key], var_name="CLOSEST_WATERDEPTH"
            )
            # check that there are no np.nans
            assert floodexpGDFs[i].isnull().sum().sum() == 0

    # merge back to large BAG dataset and add indicators for flooded areas
    finalGDF = merge_flood_datasets_to_bag(floodexpGDFs, RKdatadict, bagGDF)
    finalGDF = get_flood_status(finalGDF)
    finalGDF = get_flood_variables(finalGDF)
    finalGDF = get_conditional_waterdepth(finalGDF)
    finalGDF = get_july_floods_indicators(finalGDF)

    # compute most likely flood scenario
    finalGDF = get_most_likely_scenario(finalGDF)
    finalGDF = get_most_likely_scenario(finalGDF, extended=True)

    # keep only floodable addresses
    finalGDF = finalGDF.query("FLOOD_STATUS == 'floodable'")
    assert len(finalGDF) == 36124

    # if RVO data not present: assign flood status randomly
    if "FLOODED" not in finalGDF.columns:

        rows_to_assign = finalGDF.sample(n=2686 , replace=False).index
        finalGDF.loc[rows_to_assign, 'FLOODED'] = 1
        finalGDF["FLOODED"] = finalGDF["FLOODED"].fillna(0)

    # add flood status indicator variables
    finalGDF = add_floodmaps_indicators(
        finalGDF, "WATERDEPTH_MAX", "FLOOD_MAX"
        )

    # remove ``FLOOD_MAX == 10'', which drops 91 observations (effective full sample is 36033)
    finalGDF = finalGDF.query("FLOOD_MAX_10 != 1").copy()
    assert len(finalGDF) == 36033

    finalDF = finalGDF.drop(columns=["geometry"])
    finalDF.to_csv(produces, sep=";")