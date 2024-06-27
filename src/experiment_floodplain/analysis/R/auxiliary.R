# Functions to perform Rigorous Lasso (`hdm` package).

library(glue)
library(tidyr)
library(devtools)

#' Minimal cleaning of dataframe.
#'
#' @param df Dataframe of interest. 
#' Need to contain columns "wtp_info", "correct_waterdepth", and "treatment".
#'
#' @return Clean dataframe.
data_housekeeping <- function(df){
  # replacing values in wtp for information
  df <- df %>% 
    mutate(wtp_info = replace(wtp_info, wtp_info == -99, NA))
  
  # replacing values in column "correct waterdepth" 
  df <- df %>% 
    mutate(correct_waterdepth = replace(
      correct_waterdepth,
      correct_waterdepth == "between 2 and 5m" | correct_waterdepth == "more than 5m",
      "more than 2m"))
  
  # create dummy variables for treatment
  df <- fastDummies::dummy_cols(df, "treatment")
  
  return(df)
}


#' Drop missing values from dataframe
drop_missing_values <- function(df, outcome, covariates){
  # only keep subset of data without missing values  
  data <- subset(df, (!is.na(df[[outcome]])))
  for (cov in covariates){
    data <- subset(data, (!is.na(data[[cov]])))
  }
  
  return(data)
}


#' Run weighted least square.
#'
#' @param df Dataframe of interest
#' @param outcome Dependent variable
#' @param covariates Matrix of covariates
#' @param treatment (Binary) treatment variable(s)
#'
#' @return Dataframe of results.
run_wls <- function(
  data, outcome, covariates, treatment){
  # vector of weights
  weights <- data$WEIGHTS
  
  # fit weighted least square
  regressors <- c(treatment, covariates)
  formula <- formula(sprintf(
    "wtp_insurance_wins975 ~ %s - 1", 
    paste(regressors, collapse=" + ")))
  fit_object <- lm(
    data = data, formula=formula, x=TRUE, y=TRUE, weights=weights)
  
  # return fitted model
  return(fit_object)
}


#' Run rigorous lasso from `hdm` package.
#'
#' @param data Dataframe of interest
#' @param y_col Dependent variable
#' @param d_col Treatment variable (binary)
#' @param x_cols Matrix of covariates
#' @param weights_col Sampling weights
#'
#' @return Dataframe of results.
single_rlasso <- function(
    data, y_col, d_col, x_cols, weights_col) {
  # only keep subset of data with non-nan in outcome
  data <- subset(data, (!is.na(data[[y_col]])))
  # covariates
  X = data[, c(x_cols)]
  X = as.matrix(X)
  # outcome
  y = data[[y_col]]
  # sample weights
  weights = data[[weights_col]]
  # treatment column
  d = data[[d_col]]
  # run post-lasso
  dX = as.matrix(cbind(d, X))
  postlasso.effect = rlassoEffect(
    x = X,
    y = y,
    d = d,
    weights=weights,
    method = "partialling out"
  )
  # run double selection
  dblasso.effect = rlassoEffect(
    x = X,
    y = y,
    d = d,
    weights=weights,
    method = "double selection"
  )
  # bind all results
  res <- rbind(
    summary(postlasso.effect)$coef[1,],
    summary(dblasso.effect)$coef[1,]
  )
  # assign names
  rownames(res) = c("Post-lasso", "Double selection")
  return(res)
}