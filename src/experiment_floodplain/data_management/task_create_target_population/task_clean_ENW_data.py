"""This script cleans the raw data in *bld/replication_data/ENW*, creates a GeoDataFrame
and saves it as geopackage (.GPKG) in *bld/data/ENW*.

The final dataset contains four columns:

    - ``FEATURE``, indicating the event type (incident, evacuation, or flood);
    - ``geometry``, containing the shapely Polygons or shapely Point associated
      with the ``FEATURE``;
    - ``Status``, which is always 1 except for some evacuations (as some areas
      were not formally evacuated);
    - ``COMMENT`` containing additional information (e.g. incident acronym as
      found in ENW " Hoogwater 2021 Feiten en Duiding" report).

"""
import pytask
import numpy as np
import pandas as pd
import geopandas as gpd

from experiment_floodplain.config import SRC, BLD

ENWpath = BLD / "replication_data" / "ENW" / "floods-july-2021"
depends_on = {
    "geul": ENWpath / "floodsGeul" / "GeulFloodExtent.shp",
    "maas": ENWpath / "floodsMaas" / "MaasFloodExtent.shp",
    "roer": ENWpath / "floodsRoer" / "RoerFloodExtent.shp",
    "incidents": ENWpath / "incidents" / "incidents.shp",
    "evacuations": ENWpath / "evacuations" / "evacuations.shp",
}
produces = {"ENW": BLD / "data" / "ENW" / "ENW.GPKG"}


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_clean_ENW_data(depends_on, produces):
    # load data
    ENWlist = {}
    for key in depends_on.keys():
        ENWlist[key] = gpd.read_file(depends_on[key])

    # clean "incidents" GeoDataFrame
    incidentsGDF = ENWlist["incidents"].copy().rename(columns={"Beschrijvi": "COMMENT"})
    incidentsGDF["COMMENT"] = incidentsGDF.COMMENT.str[-5:-1]
    incidentsGDF["COMMENT"] = incidentsGDF.COMMENT.str.replace("(", "", regex=False)
    incidentsGDF["FEATURE"] = "incident"
    incidentsGDF["STATUS"] = 1

    # clean "evacuations" dataset
    evacuationsGDF = ENWlist["evacuations"].to_crs(
        "epsg:28992"
    )  # for some reason the evacuation dataset has a different crs
    evacuationsGDF["FEATURE"] = "evacuation"
    evacuationsGDF = evacuationsGDF.rename(columns={"Status": "COMMENT"}).drop(
        columns=["id", "Name"]
    )
    evacuationsGDF["COMMENT"] = evacuationsGDF["COMMENT"].replace(
        {
            "Noodverordening zonder evacuatie": "Emergency ordinance without evacuation",
            "Evacuated (ook na 19 Juli)": "Evacuated",
            "Afgesloten vanwege slib": "Closed due to sludge",
            "Evacuated (advies politie)": "Evacuated",
            "Stroomuitval": "Power failure",
        }
    )
    evacuationsGDF["STATUS"] = np.where(
        evacuationsGDF.COMMENT == "Evacuated", 1, 0
    )  # define "evacuated geometry" as any geometry with EVACUATION_STATUS = 'Evacuated'

    # clean "geul" dataset
    geulGDF = ENWlist["geul"].set_crs(
        "epsg:28992"
    )  # set coordinates to Geul dataset (they are missing in the raw data)
    geulGDF = geulGDF.rename(columns={"wdp": "COMMENT"}).drop(columns=["DN"])
    geulGDF["FEATURE"] = "Geul flood"
    geulGDF["STATUS"] = 1
    geulGDF["COMMENT"] = geulGDF["COMMENT"].replace(
        {0.5: "Estimated water depth: 0.5m", 0.0: "Estimated water depth: 0.0m"}
    )

    # clean "maas" dataset
    maasGDF = ENWlist["maas"]
    maasGDF["COMMENT"] = "-"
    maasGDF["STATUS"] = 1
    maasGDF["FEATURE"] = "Maas flood"
    maasGDF = maasGDF[["COMMENT", "geometry", "STATUS", "FEATURE"]]

    # clean "roer" dataset
    roerGDF = ENWlist["roer"].rename(columns={"fid": "COMMENT"})
    roerGDF["COMMENT"] = roerGDF["COMMENT"].replace(
        {
            152.0: "Estimated water depth: 0.152m",
            91.0: "Estimated water depth: 0.91m",
            293.0: "Estimated water depth: 0.293m",
        }
    )
    roerGDF["STATUS"] = 1
    roerGDF["FEATURE"] = "Roer flood"
    roerGDF = roerGDF[["COMMENT", "geometry", "STATUS", "FEATURE"]]

    # concatenate GeoDataFrames and saves result
    EnwGDF = pd.concat([maasGDF, roerGDF, geulGDF, incidentsGDF, evacuationsGDF])
    EnwGDF["COMMENT"] = EnwGDF.COMMENT.replace({"-": "None"})
    EnwGDF.to_file(produces["ENW"], driver="GPKG")
