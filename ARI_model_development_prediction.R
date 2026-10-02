
#Load the packages
install.packages("xgboost", repos = "https://p3m.dev/cran/2025-12-01") # An older version of xgboost must be installed, to work with parallel processing within caret. 
library(caret)
library(xgboost)
library(pROC)
library(dplyr)
library(parallel)
library(doParallel)




ari_bip <- ari_model_evaluation(
  data=example_bip, #The name of the dataframe/dataset
  outcome="bip", #The name of the outcome - NOTE how it's written "bip" and not bip
  id = "id", #The name of the ID variable
  exclude_vars = c("sex"), #This option is used for excluding unwated variables in the dataset. Set to NULL if 
                              #your data only contains variables you want to predict with. 
  seed = 242323, #Setting seed for reproducabiity
  train_fraction = 0.8, #The proportion if the data to be used for training - 80% is standard. 
  n_cores = 1 #Numbers of cores used. The more cores the faster the program, but too many cores might overload 
              #the CPU/memory. Optimal number of cores depends on size of dataset and CPU/memory capability. 
)

ari_bip$summary #Summarizes the AUC and computation speed, while comparing it to logistic regression. 
#Note that below is built upon randomly simulated data. 
# ari_bip$summary
#     Model   AUC Runtime_Minutes
# Logistic 0.519            0.00
#  ARI_XGB 0.508            0.94



ari_pred <- ari_prediction(
    data=example_bip, #The name of the dataframe/dataset
    outcome="bip", #The name of the outcome - NOTE how it's written "bip" and not bip
    id = "id", #The name of the ID variable
    birthyear="birthyear", #The name of the birthyear variable
    exclude_vars = c("sex"), #This option is used for excluding unwated variables in the dataset. Set to NULL if 
                              #your data only contains variables you want to predict with. 
    seed = 242323, #Setting seed for reproducabiity
    n_folds = 5, #The number of folds - 5 is usually the standard. 
    n_cores=1 #Numbers of cores used. The more cores the faster the program, but too many cores might overload 
              #the CPU/memory. Optimal number of cores depends on size of dataset and CPU/memory capability. 
)

ari_pred #This dataframe will contain 5 variables
#1. The ID variable
#2. ari_prediction - the reported probability for each individual. 
#3. ari_z the birthyear-standardized ARI score. 
#4. ari_decile the score-dependent 1-10 decile, with 1 being the lowest and 10 the highest. 
#5. ari_strata similar to the decile, but the highest decile is made into centiles (10.1-10.10).






