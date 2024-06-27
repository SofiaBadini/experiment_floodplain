"""This script cleans the raw data in *bld/replication_data/BAG*, creates a DataFrame and
saves it as .csv file in *bld/data/BAG*.

The final dataset contains full addresses and some building-related characteristics
for all (in-between, detached, semi-detached, corner or two-under-one-roof) houses
in Limburg that, as of 2022:

    - Belong to buildings "in use";
    - Are used for residential purposes;
    - Have a one-to-one relationship with their building (e.g. no houses that
      share the same building with shops or other spaces);
    - Have unique addresses based on street, housenumber, and postcode.

These restrictions are supposed to facilitate reaching our sample via survey
invitation letters and to avoid ambiguities on the effective flood exposure of
the survey respondents (for example, I drop apartments because it is not possible
to tell on which floor they are located).

"""
import pytask
import pandas as pd
import geopandas as gpd
import json

from tqdm import tqdm

from experiment_floodplain.config import SRC, BLD
from experiment_floodplain.data_management.task_create_target_population.merge_data import load_BAG_chunk

depends_on = {
    "addresses": BLD / "replication_data" / "BAG" / "bag-adressen-woning-nl.csv.zip",
    "bag_labels": SRC / "data_management" / "task_create_target_population" / "json" / "bagLabels.json"
}
produces = {"BAG": BLD / "data" / "BAG" / "BAG.csv"}


@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_clean_BAG_data(depends_on, produces):
    BAGdict = json.load(open(depends_on["bag_labels"]))
    # technical info: this dataset has 9,576,996 rows
    chunks = list(range(1_000_000, 10_000_000, 1_000_000))
    addDFs = [
        load_BAG_chunk(depends_on["addresses"], BAGdict, chunk)
        for chunk in tqdm(chunks)
    ]
    addDF = pd.concat(addDFs)

    # Only keep building in use ("Pand in gebruik")
    addDF = addDF.query("BUILDINGSTATUS == 'Pand in gebruik'").copy()

    # Only keep addresses for residential functions
    addDF = addDF.query("FUNRESIDENTIAL == 1").copy()
    addDF["HOUSINGTYPE"] = addDF["HOUSINGTYPE"].replace(
        {
            "Appartement": "Apartment",
            "Tussen of geschakelde woning": "In-between or semi-detached house",
            "Vrijstaande woning": "Detached house",
            "Hoekwoning": "Corner house",
            "Tweeonder1kap": "Two-under-one-roof house",
        }
    )

    # Only keep houses (no apartments), this still leaves 393,635 observations
    addDF = addDF.query("HOUSINGTYPE != 'Apartment'")

    # Only keep addresses for which there is one-to-one relationship between house and building
    addDF = addDF[addDF["1RESIDENCE_1BUILDING"] == 1.0]

    # Drop addresses that are not unique on street, housenumber, and postcode (no houses with addition)
    # This still leaves 358,131 observations
    addDF = addDF.query("IS_PCHN_UNIQUE == 1")

    # Some renaming to facilitate future merges
    addDF = addDF.rename(columns={"POSTCODE_2022": "PC6_2022"})
    addDF["PC5_2022"] = addDF.PC6_2022.str[:-1]
    addDF["PC4_2022"] = addDF.PC5_2022.str[:-1]

    # Reset index and save
    addDF = addDF.reset_index(drop=True)
    addDF.to_csv(produces["BAG"])
