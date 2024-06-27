.. _data_management:

***************
Data management
***************

Documentation of the code in *experiment_floodplain/data_management*.
The data management process consists of three tasks:

- **Task 1**: In the folder *task_download_data*. Downloads the data necessary
  to replicate the project.

- **Task 2**: In the folder *task_create_target_population*.  Creates the target
  population, i.e., creates a dataset of the addresses that were eligible for being
  contacted (what is called "full sample" throughout the paper). The result is
  the dataset ``target_population.csv`` in *bld/data*.

- **Task 3**: In the folder *task_sample_survey_recipients*. Creates a dataset of
  the addresses that were in fact contacted. The result is the dataset
  ``survey_recipients.csv`` in *bld/data*.

  .. note::

    The dataset ``survey_recipients.csv`` needs ``rvo_pc6.csv`` to be reproduced.
    The latter will be shared upon publication of this paper. By default, this
    project creates a slightly different version of ``survey_recipients.csv`` to
    illustrate how the code works. I provide the (anonymized) dataset of the
    people who were in fact contacted under *bld/replication_data/SURVEY*.

- **Task 4**: In the folder *task_clean_survey_data*. Cleans the original Qualtrics
  survey data. The result is the dataset ``survey_data.csv`` in *bld/data*.

  .. note::

    Because of data protection concerns, the original dataset  --which is anonymized
    and contains only a subset of the original Qualtrics data-- will be shared upon
    publication of this paper. By default, this replication package
    uses fake data created based on the original dataset with the Python library
    Synthetic Data Vault (SDV). See also the Introduction of this
    replication package.

Script for Task 1: downloading the data
---------------------------------------

Script ``task_download_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_download_data.task_download_data
    :members:

Script ``generate_synthetic_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In folder *generate_synthetic_data*.
.. automodule:: experiment_floodplain.data_management.task_download_data.generate_synthetic_data.generate_synthetic_data
    :members:


Scripts for Task 2: creating the target population
--------------------------------------------------

Script ``task_clean_BAG_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_create_target_population.task_clean_BAG_data
    :members:

Script ``task_clean_ENW_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_create_target_population.task_clean_ENW_data
    :members:

Script ``task_assign_flood_exposure.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_create_target_population.task_assign_flood_exposure
    :members:


Scripts for Task 3: sampling the survey recipients
--------------------------------------------------

Script ``task_add_survey_weights_to_full_sample.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_sample_survey_recipients.task_add_survey_weights_to_full_sample
    :members:

Script ``task_create_pilot_sample.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_sample_survey_recipients.task_create_pilot_sample
    :members:

Script ``task_create_main_sample.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_sample_survey_recipients.task_create_main_sample
    :members:

Script ``task_create_survey_recipients_dataset.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_sample_survey_recipients.task_create_survey_recipients_dataset
    :members:


Scripts for Task 4: cleaning the survey data
--------------------------------------------

Sript ``task_clean_survey_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.data_management.task_clean_survey_data.task_clean_survey_data
    :members:


Auxiliary modules
-----------------

Script ``merge_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~
In the folder *task_create_target_population*.

.. automodule:: experiment_floodplain.data_management.task_create_target_population.merge_data
    :members:


Script ``sample.py``
~~~~~~~~~~~~~~~~~~~~
In the folder *task_sample_survey_recipients*.

.. automodule:: experiment_floodplain.data_management.task_sample_survey_recipients.sample
    :members:

Script ``clean_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~
In the folder *task_clean_survey_data*.

.. automodule:: experiment_floodplain.data_management.task_clean_survey_data.clean_data
    :members:
