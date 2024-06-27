.. _introduction:


************
Introduction
************

------------
This project
------------

This is the documentation for the replication package of the paper "Information
frictions, overconfidence, and learning: Experimental evidence from a floodplain"
(`PDF  <https://drive.google.com/file/d/12N7N-KCTPBidlzxtHDkb5e8cirPAoJeh/view?usp=sharing>`_,
`pre-analysis plan <https://osf.io/yxc3m>`_), by `Sofia Badini <https://sofiabadini.github.io/>`_.

**Abstract**. I use an online experiment to study whether offering information
to floodplain residents is sufficient to change their perceived risk exposure
and demand for insurance. The participants are offered information on the flood
risk profile at their address and on the rules over compensation of flood damages.
I find that respondents tend to misperceive their risk category according to
publicly available flood maps, but express high levels of confidence in their
guesses. When not prompted to engage with the information they are offered, one
third of them read nothing. Respondents who are asked to read information on their
risk profile tend to stop reading any further and report a lower willingness-to-pay
for insurance. However, this effect does not seem to be driven by respondents
learning more from the information they are provided with, at least based on how
they update their beliefs. Instead, I find suggestive evidence of backlash to
information among residents of high risk areas and individuals who initially
underestimated their risk category.

------------------------
Structure of the project
------------------------

This project uses the `econ-project-templates <https://econ-project-templates.readthedocs.io/en/stable/>`_
by Hans-Martin von Gaudecker, which in turn use `pytask <https://pytask-dev.readthedocs.io/en/stable/>`_,
a workflow management system for reproducible data analysis. Therefore, this project is fully and
automatically reproducible conditional on having
`conda <https://docs.conda.io/en/latest/>`_, a modern LaTeX distribution
(e.g. `MikTex <https://miktex.org/>`_), and the text editor
`VS Code <https://code.visualstudio.com/>`_ installed. You will
also need to install `R <https://www.r-project.org/>`_. Please see `the documentation
of econ-project-templates <https://econ-project-templates.readthedocs.io/en/stable/getting_started/index.html#preparing-your-system>`_ for
an overview of the dependencies that you need to install (and for detailed explanations
on why the template looks the way it does).

The minimum information that you need to know is the following:

 - The important scripts in this project are stored in the folder *src*,
   specifically in the sub-folders *experiment_floodplain/data_management*,
   *experiment_floodplain/analysis*, and *experiment_floodplain/final*.
 - Upon replication of the project, a new folder named *bld* will appear in the
   root folder (where *src* is). *bld* contains all the output of this project.
 - *bld* also contains this website, under the sub-folder *documentation*. The
   documentation is also produced in PDF version: it is the file ``documentation.pdf``
   (under *bld/documentation/latex*, a copy is in the root folder).
 - **The single most important file produced by this replication package is the
   .pdf** ``paper_results_replication.pdf``, which reproduces all the figures and
   tables in the paper (under *bld/latex*, a copy will appear in the root folder).

------------------------
Issues with data sharing
------------------------

This project uses survey data collected with the platform `Qualtrics <https://www.qualtrics.com/>`_.
The survey respondents were contacted via postcards mailed to their addresses, as the
Dutch government maintains a `database <https://business.gov.nl/regulation/addresses-and-buildings-key-geo-register/>`_
of publicly available geocoded addresses for the entire country. Each postcard included a unique code,
which served both as a password to log-in to the online survey, and as an identifier (survey respondents
were displayed address-specific information). I additionally determine whether each address was flooded in July 2021
using a dataset of postcode-level data provided by the government (which is not
available to the public).

These two datasets will be shared once this paper is published, and will be delivered
as a dataset in .csv format (named ``original_survey_data.csv`` and ``rvo_pc6.csv``).
The original data collected via Qualtrics will be anonymized and will include only
the subset of the original Qualtrics data that I used for the analysis.

.. note::

  **The analysis will be automatically performed on the real data if the .csv files
  are placed in the folder** *src/experiment_floodplain/data*. By default, this
  replication package uses fake data, based on the original dataset and created with
  the Python library `Synthetic Data Vault (SDV) <https://sdv.dev/>`_, in place of
  the real survey data. Moreover, by default the flood status is attributed at
  random. See also the section Data Management.

-----------
Replication
-----------

To reproduce, first navigate to the root of the project (where *src* is). Then,
open your terminal emulator and run, line by line::

    conda create --name condalock python # create empty environment
    conda activate condalock # activate environment
    pip install conda-lock # install conda-lock

The lines above install `conda-lock <https://conda.github.io/conda-lock/>`_ in a
clean environment. conda-lock is a Python package to generate reproducible conda
environments -- i.e., directories containing the packages needed to run code.
To install the environment for this project, after the lines above you additionally
need to run::

    conda-lock install --name experiment_floodplain environment.conda-lock.yml # create project environment with conda-lock
    conda activate experiment_floodplain # activate project environment

Finally, to execute the project run::

    pip install -e . # install the project
    pytask # execute the project

If your machine runs Windows11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... you may want to use the file ``environment_win11.conda-lock.yml`` instead of
``environment.conda-lock.yml``. The former file uses an older version of ``pytask``,
as the newest releases appear to... behave capriciously with Windows11.
