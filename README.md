# Replication package for "Information Frictions, Overconfidence, and Learning: Experimental Evidence from a Floodplain"

## This project

This is replication package of the paper "Information
frictions, overconfidence, and learning: Experimental evidence from a floodplain" ([PDF](https://drive.google.com/file/d/12N7N-KCTPBidlzxtHDkb5e8cirPAoJeh/view?usp=sharing) | [Pre-analysis plan](https://osf.io/yxc3m) | [Replication data](https://data.4tu.nl/datasets/46fa1840-6522-45cd-b36c-038b259c4c95)).

**Abstract.** I use an online experiment to study whether offering information to floodplain residents is sufficient to change their perceived risk exposure and demand for insurance. The participants are offered information on the flood risk profile at their address and on the rules over compensation of flood damages. I find that respondents tend to misperceive their risk category according to publicly available flood maps, but express high levels of confidence in their guesses. When not prompted to engage with the information they are offered, one third of them read nothing. Respondents who are asked to read information on their risk profile tend to stop reading any further and report a lower willingness-to-pay for insurance. However, this effect does not seem to be driven by respondents learning more from the information they are provided with, at least based on how they update their beliefs. Instead, I find suggestive evidence of backlash to information among residents of high risk areas and individuals who initially underestimated their risk category.


## Structure of the project

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates). It uses [pytask](https://pytask-dev.readthedocs.io/en/stable/)
a workflow management system for reproducible data analysis.
Therefore, this project is fully and
automatically reproducible conditional on having
[conda](https://docs.conda.io/en/latest/), a modern LaTeX distribution
(e.g. [MikTex](https://miktex.org/)), and the text editor [VS Code](https://code.visualstudio.com/) installed. You will
also need to install [R](https://www.r-project.org/).


Please see the [documentation
of econ-project-templates](https://econ-project-templates.readthedocs.io/en/stable/getting_started/index.html#preparing-your-system) for
an overview of the dependencies that you need to install and for detailed explanations
on why the template looks the way it does. The minimum information that you need to know is the following:

 - The important scripts in this project are stored in the folder *src*, specifically in the sub-folders *experiment_floodplain/data_management*, *experiment_floodplain/analysis*, and *experiment_floodplain/final*.
 - Upon replication of the project, a new folder named *bld* will appear in the root folder (where *src* is). *bld* contains all the output of this project.
 - The single most important file in *bld* is the .pdf     ``paper_results_replication.pdf``, which reproduces all the figures and tables in the paper (under *bld/latex*, but a copy will appear in the root folder).

## Issues with data sharing

This project uses survey data collected with the platform [Qualtrics](https://www.qualtrics.com/). The survey respondents were contacted via postcards mailed to their addresses, as the
Dutch government maintains a [database of publicly available geocoded addresses](https://business.gov.nl/regulation/addresses-and-buildings-key-geo-register/) for the entire country. I additionally determine whether each address was flooded in July 2021 using a dataset of postcode-level data provided by the government (which is not available to the public).

These two datasets will be shared once this paper is published, and will be delivered as a dataset in .csv format (named ``original_survey_data.csv`` and ``rvo_pc6.csv``).
The original data collected via Qualtrics will be anonymized and will include only the subset of the original Qualtrics data that I used for the analysis.

The analysis will be automatically performed on the real data if the .csv files are placed in the folder *src/experiment_floodplain/data*. **By default, this replication package uses synthetic data**, based on the original dataset and created with the Python library [Synthetic Data Vault (SDV)](https://sdv.dev/>), in place of the real survey data. Moreover, by default the flood status is attributed at random. For more information, see the Readme of the [replication data](https://data.4tu.nl/datasets/46fa1840-6522-45cd-b36c-038b259c4c95) hosted on 4TU.ResearchData, plus the section Data Management of the Read the Docs documentation.

## Usage

To reproduce, first navigate to the root of the project (where *src* is). Then, open your terminal emulator and run, line by line:

```console
$ conda create --name condalock python
$ conda activate condalock
$ pip install conda-lock
```

The lines above install [conda-lock](https://conda.github.io/conda-lock/) in a clean environment. conda-lock is a Python package to generate reproducible conda environments -- i.e., directories containing the packages needed to run code.

To install the environment for this project, after the lines above you additionally need to run:

```console
$ conda-lock install --name experiment_floodplain environment.conda-lock.yml
$ conda activate experiment_floodplain
```

Finally, to execute the project run:

```console
$ pip install -e .
$ pytask
```

### If your machine runs Windows11

... you may want to use the file ``environment_win11.conda-lock.yml`` instead of ``environment.conda-lock.yml``. The former file uses an older version of ``pytask``, as the newest releases appear to... behave capriciously with Windows11.
