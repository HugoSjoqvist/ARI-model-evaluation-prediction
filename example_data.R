# Example Data for ARI Model Evaluation and Prediction
# 
# This file contains template example data that meets the requirements
# specified in ARI_model_development_prediction.R
#
# Data Structure Requirements:
# - Each row represents one individual (index person)
# - Includes outcome variable and predictor variables only
# - Predictors: number of relatives AND number of affected relatives per relative type
# - All predictors must be numeric
# - Outcome variable must be binary (0 = unaffected, 1 = affected)
# - Short outcome names recommended (e.g., "asd", "adhd", "id", "bip")

# Example dataset for bipolar disorder prediction
# This is simulated data and should be replaced with your own data

example_bip <- data.frame(
  id = 1:100,
  birthyear = rep(1950:2000, length.out = 100),
  sex = rep(c("M", "F"), 50),
  bip = rbinom(100, 1, 0.15),  # Binary outcome: 0 = unaffected, 1 = bipolar affected
  
  # Parents
  n_parents = rep(2, 100),
  n_parents_bip = rbinom(100, 2, 0.10),
  
  # Siblings
  n_siblings = sample(0:8, 100, replace = TRUE),
  n_siblings_bip = rbinom(100, 8, 0.10),
  
  # Grandparents
  n_grandparents = rep(4, 100),
  n_grandparents_bip = rbinom(100, 4, 0.08),
  
  # Cousins
  n_cousins = sample(0:20, 100, replace = TRUE),
  n_cousins_bip = rbinom(100, 20, 0.05),
  
  # Aunts/Uncles
  n_aunts_uncles = sample(0:10, 100, replace = TRUE),
  n_aunts_uncles_bip = rbinom(100, 10, 0.12)
)

# View the structure
str(example_bip)
head(example_bip)

# Usage:
# Load this file with: source("example_data.R")
# Then use example_bip in your ari_model_evaluation() or ari_prediction() calls
