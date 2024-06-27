## This script uses the `hdm` package  to select controls for the ATE estimationg 
## via Lasso end Post-Lasso methods for for high-dimensional approximately sparse models.

install.packages("pak", repos='https://r-lib.github.io/p/pak/devel/')
library(pak)

install.packages("devtools", repos='http://cran.us.r-project.org')
install.packages("fastDummies", repos='http://cran.us.r-project.org')
install.packages("dplyr", repos = "http://cran.us.r-project.org")
install.packages("shiny", repos = "http://cran.us.r-project.org")
install.packages("tidyr", repos = "http://cran.us.r-project.org")

# The latest version of MASS as of 18/06/2024 is not compatible with R<=4.4.0,
# but r-base=4.4.0 is not available via conda. See also here:
# https://forum.posit.co/t/mass-not-available-for-r-4-3-3/188156
# Therefore, I am i`nstalling Matrix and MASS in versions that work with
# the r-base version installed in the environment, and then installing
# the `hdm` package so that the two packages are not upgraded.
devtools::install_version("Matrix", version = "1.6-5", repos = "http://cran.us.r-project.org")
devtools::install_version("MASS", version = "7.3.60.0.1", repos = "http://cran.us.r-project.org")
pak::pkg_install("hdm", upgrade=FALSE)

library(dplyr)
library(hdm)
library(fastDummies)
library(glue)

source("src/experiment_floodplain/analysis/R/auxiliary.R")

# load survey
survey_df <- read.csv('bld/data/survey_data.csv', sep = ";", header=TRUE, encoding="latin1")
survey_df <- data_housekeeping(survey_df)

# load covariates and outcomes
formulas <- read.csv('src/experiment_floodplain/analysis/csv/formulas.csv', sep=";", header=TRUE)
covariates <- formulas[formulas$VARTYPE=="PRE_COV_EXTENDED", ]$VARNAME
outcomes <- formulas[formulas$VARTYPE %in% c('OUTCOME_UPDATE', 'OUTCOME_WTP'), ]$VARNAME

# list of dataframes
data_list <- list()
for(i in list(2, 3, 4)){
  data_list[[i-1]] <- survey_df[survey_df$treatment %in% c(1, i),]
}

# list of treatments
d_col = list("treatment_2", "treatment_3", "treatment_4")

all_results <- list()

for (outcome in outcomes){
  y_col <- outcome
  # run post-lasso and double selections
  res = mapply(
    single_rlasso,
    data=data_list,
    y_col=y_col,
    d_col=d_col,
    MoreArgs = list(
      x_cols=covariates,
      weights_col="WEIGHTS"))
  # assign row names
  rownames(res) = c(
    "Estimate, post-lasso",
    "Estimate, double selection",
    "Std. Error, post-lasso",
    "Std. Error, double selection",
    "t value, post-lasso",
    "t value, double selection",
    "Pr(>|t|), post-lasso",
    "Pr(>|t|), double selection"
  )
  # assign column names
  colnames(res) = c("maps", "WTS", "insurance")
  # save results
  write.csv(
    res, glue("bld/analysis/outcomes/rlasso/rlasso_{y_col}.csv")
  )
  # store results in list
  all_results[[y_col]] <- res
}