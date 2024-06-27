.. _analysis:

********
Analysis
********

Documentation of the code in *experiment_floodplain/analysis*. The analysis consists
of three tasks:

- **Task 1**: In the folder *task_descriptive_analysis*. These scripts compute
  summary statistics over survey respondents' beliefs (``task_summary_beliefs.py``),
  information quality (``task_summary_information_frictions.py``) and reading
  behavior during the experimental part of the survey (``task_summary_reading_behavior.py``).

- **Task 2**: In the folder *task_check_experiment_randomization*. Checks for the
  integrity of the experimental randomization (``task_check_randomization.py``)
  and compare characteristics of survey respondents, survey recipients, and full
  sample (``task_check_survey_respondents.py``).

- **Task 3**: In the folder *task_estimate_treatment_effects*.


Scripts for Task 1: descriptive analysis
----------------------------------------

Script ``task_summary_beliefs.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_descriptive_analysis.task_summary_beliefs
    :members:

Script ``task_summary_information_frictions.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_descriptive_analysis.task_summary_information_frictions
    :members:

Script ``task_summary_reading_behavior.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_descriptive_analysis.task_summary_reading_behavior
    :members:


Scripts for Task 2: randomization checks
----------------------------------------

Script ``task_check_randomization.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_check_experiment_randomization.task_check_randomization
    :members:

Script ``task_check_survey_respondents.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_check_experiment_randomization.task_check_survey_respondents
    :members:


Scripts for Task 3: analysis of experimental results
----------------------------------------------------

Script ``task_estimate_ATE.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_estimate_treatment_effects.task_estimate_ATE
    :members:

Script ``task_run_rlasso.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_estimate_treatment_effects.task_run_rlasso
    :members:

Script ``task_heterogeneity_analysis.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: experiment_floodplain.analysis.task_estimate_treatment_effects.task_heterogeneity_analysis
    :members:


Auxiliary modules
-----------------

Contain functions used in the scripts listed above.

Script ``descriptive_stats.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the folder *PYTHON*.

.. automodule:: experiment_floodplain.analysis.PYTHON.descriptive_stats
    :members:

Script ``regressions.py``
~~~~~~~~~~~~~~~~~~~~~~~~~
In the folder *PYTHON*.

.. automodule:: experiment_floodplain.analysis.PYTHON.regressions
    :members:

Scripts ``auxiliary.R`` and ``run_rlasso.R``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the folder *R*.

The script ``run_rlasso.R`` uses the R package `hdm` to select
controls for the ATE estimation, via Lasso end Post-Lasso methods for high-dimensional
approximately sparse models. The script ``auxiliary.R`` contains auxiliary functions
to carry out this task.

All the functions in these scripts are duly documented in the .R files.
Unfortunately, I cannot export the docstrings with Sphinx (the software that
builds this documentation) because the R language
`is still not supported <https://github.com/sphinx-doc/sphinx/issues/10475>`_.
