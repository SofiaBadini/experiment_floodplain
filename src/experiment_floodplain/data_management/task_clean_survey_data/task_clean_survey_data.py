"""This script cleans the original Qualtrics data."""

import pytask
import pandas as pd
import numpy as np
from pathlib import Path

from functools import reduce

from experiment_floodplain.config import SRC, BLD
import experiment_floodplain.data_management.task_clean_survey_data.clean_data as clean_data

pd.set_option('future.no_silent_downcasting', True)

survey_path = BLD / "replication_data" / "SURVEY"
depends_on = {
    "survey_synthetic": survey_path / "synthetic_survey_data.csv",
    "original_variables": survey_path / "original_variables.csv",
    "final_variables": survey_path / "final_variables.csv",
    }

produces = BLD / "data" / "survey_data.csv"

@pytask.mark.depends_on(depends_on)
@pytask.mark.produces(produces)
def task_clean_survey_data(depends_on, produces):

    # load variables names and block assignment (in "raw" data)
    original_var_df = pd.read_csv(depends_on["original_variables"], sep=";")
    
    # load final variables (in final dataset used for the analysis)
    final_var_df = pd.read_csv(depends_on["final_variables"], sep=";")

    # load qualtrics data (synthetic if original dataset is not found)
    original_data = Path(SRC / "data" / "original_survey_data.csv")
    path_to_data = original_data  if original_data.is_file() else depends_on["survey_synthetic"]
    survey_df = pd.read_csv(path_to_data, sep=";", low_memory=False, index_col=[0])

    # all the "blocks" of variables
    blocks = [
        "BELIEFS",
        "INFORMATION",
        "TECHNICAL",
        "TREATMENT",
        "COVARIATES",
        "WTP"
    ]

    # all the functions to clean each block of variable
    funs = [
        clean_data.clean_beliefs_data, 
        clean_data.clean_info_data, 
        clean_data.clean_tech_data, 
        clean_data.clean_treatment_data, 
        clean_data.clean_covariates_data,
        clean_data.clean_wtp_data,
        ]
    
    # clean everything
    vars_dfs = [
        clean_data.clean_variables_block(
            survey_df, # original data
            original_var_df, # variables in original data 
            final_var_df, # variables in final (clean) data 
            block, # "block" of variables to clean
            fun # function to use
            ) 
        for block, fun in zip(blocks, funs)
        ]
    
    # merge clean dataframes on `uniqueadd_id`
    final_df = reduce(lambda df1, df2: pd.merge(
        df1, df2, on=["uniqueadd_id"]), vars_dfs
        )
    
    final_df = clean_data.add_floodmaps_indicators(
        final_df, "correct_waterdepth", "correct_floodmaps"
        )

    # add dummy variables indicating whether outcomes are present
    final_df = clean_data.add_outcomes_dummies(final_df)
    
    # add whether the attention text is passed (depends on floodmaps indicator)
    final_df = clean_data.clean_text_evaluation(final_df)

    # add information on revised beliefs
    final_df = clean_data.add_revise_beliefs_variables(final_df)

    # add information from population data
    cols_to_add = ['FLOODED', 'dist_floodareas', 'GEMNAME_2022', 'WEIGHTS', 'uniqueadd_id']
    final_df = final_df.merge(survey_df[cols_to_add], on="uniqueadd_id")

    # check all variables are there
    all_variables = final_df.columns.tolist()
    expected_variables = final_var_df.VARIABLE.tolist()
    assert set(all_variables) == set(expected_variables)

    # reorder columns and save
    final_df = final_df[expected_variables]
    final_df.to_csv(produces, sep=";", encoding="latin1", index=False)