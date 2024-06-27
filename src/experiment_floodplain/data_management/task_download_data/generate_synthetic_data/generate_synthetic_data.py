"""This script generates synthetic survey data based on the original one, using the
python package `Synthetic Data Vault`_ (this package is not part of the environment 
and needs to be installed separately, in case you want to run this script).
The resulting dataset, ``synthetic_survey_data.csv``, is downloaded to *bld/replication_data/SURVEY*.


.. note::

  This task is not part of the project workflow.

.. _Synthetic Data Vault: https://docs.sdv.dev/sdv

"""
import numpy as np
import torch
import pandas as pd
import json
import sdv

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

# (should) ensure reproducibility
np.random.seed(24)
torch.manual_seed(24)

# load real data
data = pd.read_csv("original_survey_data.csv", sep=";", index_col=[0], low_memory=False).drop(columns=["uniqueadd_id"])

# automatically generate the metadata, saved as .json to ensure reproducibility with future versions of SDV
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.save_to_json("metadata.json")

# generate synthetic data
synthesizer = GaussianCopulaSynthesizer(metadata)
synthetic_data = []
for t in [1, 2, 3, 4]:
    tdata = data.query("treatment_assignment == @t")
    synthesizer.fit(tdata)
    synthetic_df = synthesizer.sample(num_rows=len(tdata))
    synthetic_data.append(synthetic_df)

# add NaNs
tdata = data.query("treatment_assignment.isna()")
synthesizer.fit(tdata)
synthetic_df = synthesizer.sample(num_rows=len(tdata))
synthetic_data.append(synthetic_df)

# concat
synthetic_data = pd.concat(synthetic_data)
synthetic_data = synthetic_data.reset_index(drop=True)

# Data Validity and Data Structure: 100%
diagnostic = run_diagnostic(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# only matters for where we have treatment status
# Column shapes around 89%, Column Pair Trends around 86%
for t in [1, 2, 3, 4]:
    quality_report = evaluate_quality(
        data.query("treatment_assignment == @t"),
        synthetic_data.query("treatment_assignment == @t"),
        metadata
    )

# add unique id
synthetic_data["uniqueadd_id"] = synthetic_data.index

# too few 0 WTP in synthetic data, need to add some
zero_wtp_df = (data
    .query("insurance_wtp == 0")
    .groupby("treatment_assignment")
    .insurance_wtp.count()
    .to_frame()
)
for t in [1, 2, 3, 4]:
    # count how many 0 wtp under each treatment arm
    n = (zero_wtp_df
            .query("treatment_assignment == @t")
            .insurance_wtp.values[0]
    )
    # get index of rows where zero wtp needs to be imputed
    zero_wtp_index = (synthetic_data
        .loc[synthetic_data["treatment_assignment"] == t, ["insurance_wtp"]]
        .sort_values(by="insurance_wtp", ascending=True)
        .iloc[:n]
    ).index 
    # impute wtp
    synthetic_data.loc[zero_wtp_index, "insurance_wtp"] = 0

# save
synthetic_data.to_csv("synthetic_survey_data.csv", sep=";")