"""This script adjusts the survey recipients dataset to reflect the addresses
that were in fact contacted *after* data collection. In particular:

    - Add indicator variable for incorrect addresses (where mail was rejected).
    - Add indicator variable for whether the address was reached during data collection,
      and at what stage.

The resulting dataset is saved as `survey_recipients.csv` in *bld/data*.

.. note::
   This dataset is identical to `survey_recipients.csv` in *bld/replication_data/SURVEY*
   only if the dataset *rvo_pc6.csv* is placed under *src/experiment_floodplain/data*.

"""
import pytask
import pandas as pd
import numpy as np

from pathlib import Path
from experiment_floodplain.config import SRC, BLD

depends_on = {
    "fullSample": BLD / "data" / "full_sample.csv",
    # survey recipients
    "pilot": BLD / "data" / "survey_recipients" / "pilot" / "pilot_sample.csv",
    "main": BLD / "data" / "survey_recipients" / "main" / "main_sample.csv",
    # returned postcards
    "returned": SRC / "data_management" / "task_sample_survey_recipients" / "csv" / "returnedPostcards.csv",
}

produces = BLD / "data" / "survey_recipients.csv"

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_create_full_survey_sample_with_weights(depends_on, produces):

    # contacted addresses
    dfs = []
    for p in ["pilot", "main"]:
        df = pd.read_csv(
            depends_on[p],
            sep=";",
            encoding="latin1",
            low_memory=False
            )
        df["data_collection_stage"] = p
        dfs.append(df)
    
    surveyRecipients = pd.concat(dfs)

    # remove addresses from which postcards came back.
    # only relevant with data on real flood status 
    # (otherwise the sampling will be slightly off, thus
    # these addresses may not be in the data.)
    path_to_rvo_data = Path(SRC / "data" / "rvo_pc6.csv")
    if path_to_rvo_data.is_file():

        # postcards that came back
        returnedPostcards = pd.read_csv(depends_on["returned"], sep=";")
        returnedPostcards["returned"] = 1

        # merge returned postcards to full sample
        address_left = ['STREETNAME_2022', 'HOUSENUM_2022', 'PC6_2022', 'CITYNAME_2022']
        address_right = ['STRAATNAAM', 'HUISNUMMER', 'POSTCODE', 'PLAATSNAAM']
        surveyRecipients = surveyRecipients.merge(
            returnedPostcards,
            left_on=address_left,
            right_on=address_right,
            how="left"
            )
        surveyRecipients.returned = surveyRecipients.returned.fillna(0)
        surveyRecipients = surveyRecipients.query("returned == 0")

        # check length is correct
        assert len(surveyRecipients) == 16977 

    # keep only relevant info
    surveyRecipients = surveyRecipients[[
        'waterdepth_max_less_than_05m',
        'waterdepth_max_between_05_and_1m', 
        'waterdepth_max_between_1_and_15m',
        'waterdepth_max_between_15_and_2m', 
        'waterdepth_max_between_2_and_5m',
        'waterdepth_max_more_than_5m', 
        'flood_max_1_in_100_years',
        'flood_max_1_in_1000_years', 
        'flood_max_1_in_10000_years', 
        'FLOODED',
        'WEIGHTS'
    ]]

    # save
    surveyRecipients.to_csv(produces, sep=";", encoding="latin1")
