

############################################################
######################## README ############################
############################################################

# This script estimates an Affected Relative Index (ARI) score from
# family-history information and compares multiple machine
# learning models for prediction.
# Implemented models:
# - Logistic Regression
# - Elastic Net
# - XGBoost
# - Artificial Neural Network
# - Linear Discriminant Analysis

# Data requirements:
#
# 1. Each row should represent one individual (index person).
#
# 2. Include only:
#    - the outcome variable
#    - predictor variables intended for the model
#
# 3. Predictors should reflect both:
#    a) the number of relatives of a given type
#    b) the number of affected relatives
#
# Example:
#   n_cousins       = total number of cousins
#   n_cousins_adhd  = number of cousins with ADHD
#
# 4. All predictor variables must be numeric.
#
# 5. The outcome variable must be binary:
#    0 = unaffected
#    1 = affected
#
# 6. Short outcome names are recommended
#    (e.g. "asd", "adhd", "id").
#
# 7. Different classes of relatives and family diagnoses
#    may be used depending on the research question.
############################################################


#Setting the default pathway to the folder
setwd("P:/user/ARI_model")
#For simplicity, it is recommended to have a "data" folder containing the datasets there. 


required_packages <- c("haven","caret","glmnet","pROC","dplyr","ggplot2", "doParallel")
missing_packages <- required_packages[
  !required_packages %in% installed.packages()[, "Package"]
]
if(length(missing_packages) > 0){
  install.packages(missing_packages)
}

if (!dir.exists("output")) {
  dir.create("output")
}
if (!dir.exists("data")) {
  stop("A 'data' folder containing input datasets is required.")
}

# An older version of xgboost must be installed, to work with parallel processing
# within caret. 
install.packages("xgboost", repos = "https://p3m.dev/cran/2025-12-01")
library(haven)
library(caret)
library(glmnet)
library(xgboost)
library(pROC)
library(dplyr)
library(ggplot2)
library(parallel)
library(doParallel)


############################################################
# User settings
############################################################
seed <- 242323
birthyear_variable <- "year" #The variable which indicates the birthyear of the individuals.
id_variable <- "LopNr" #The variable which identifies the individuals
exclude_vars <- c("n_rFar", "n_rMor", "n_grandparents", "Kon") #This is the step where you decide to exclude specific variables from the data.
    #We removed sex ("Kon") from the analysis, and missing status of fathers or mothers (since there weren't any missing)
n_folds <- 5 #This decides the number of folds - 5 is usually the most common option. 

#For faster computation, we have enabled parallel processing. 
#NOTE: Due to large datasize and limited computational power, restricting cl <- 1 might often be necessary. 
  #If the program frequently crash with multiple cores, it usually indicates memory overload and should be set to n_cores=1. 
available_cores <- detectCores() #This command shows how many available cores your computer have. 
print(paste("Available cores:", available_cores))
n_cores <- 6   # Number of cores being used, change to more than 1 if parallel processing is desired



#Load in the data - NOTE: Make sure that the name corresponds exactly to the name in the actual outcome variable. 
datasets <- list(
  asd  = read_dta("data/i_asd_full_sample.dta"), #This assumes the data is a Stata-file - if other, it needs to be updated accordingly. 
  adhd = read_dta("data/i_adhd_full_sample.dta"),
  id   = read_dta("data/i_id_full_sample.dta")
)

###TEMP SAMPLE 100K FROM IT###
datasets <- lapply(datasets, function(x) {
  x[sample(nrow(x), min(300000, nrow(x))), ]
})


for (outco in names(datasets)) {
# Results are reproducible because a fixed random seed is used.
#Set seed when dividing the data into training and test set for model evaluations. 
set.seed(seed)
data_choice <- datasets[[outco]]
exclude_vars_model <- c(id_variable, exclude_vars)
data <- data_choice[, !(names(data_choice) %in% exclude_vars_model)]

data$psych_out <- data[[outco]]
data[[outco]]<-NULL
# Ensure all predictors are numeric
predictor_vars <- setdiff(names(data), "psych_out")
data[predictor_vars] <- lapply(data[predictor_vars], as.numeric)


#Here 80% of the population will be used for training, and the remaining 20% as test/evaluation. 
samp <- sample(nrow(data), 0.8 * nrow(data))
train <- data[samp, ]
test <- data[-samp, ]

#########################
#########################
#####   ML models   #####
#########################
#########################

###Logistic regression###
start.time <- Sys.time()
  logit <- glm(as.factor(psych_out) ~ ., family = binomial(link="logit"),  data=train)
  test$logit_pred <- predict(logit, test, type="response")
end.time <- Sys.time()
time_logit <- end.time - start.time
print("time_logit")
print(time_logit)

# caret expects factor levels to be valid variable names
train$psych_out <- factor(
  ifelse(train$psych_out == 1, "case", "control")
)

if(n_cores > 1){
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
}

cv <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE)

# Train the Elastic Net model
start.time <- Sys.time()
tuneGrid <- expand.grid(alpha = 0:1, lambda = seq(0.001, 0.1, by = 0.001))
model_en <- train(as.factor(psych_out) ~ ., data = train,
                  method = "glmnet",
                  preProcess = c("scale", "center"),
                  trControl = cv,
                  tuneGrid = tuneGrid,
                  allowParallel = (n_cores > 1),
                  maxit = 10000)
test$en_pred <- predict(model_en, test, type="prob")
end.time <- Sys.time()
time_en <- end.time - start.time
print("time_en")
print(time_en)

###X-gradient Boosting###
start.time <- Sys.time()
model_xgb <- train(as.factor(psych_out)~., data = train, 
                   method = "xgbTree",
                   preProcess = c("scale", "center"),
                   allowParallel = (n_cores > 1),
                   trControl = cv,
                   verbose = FALSE,
                   verbosity = 0)
test$xgb_pred <- predict(model_xgb, test, type="prob")
end.time <- Sys.time()
time_xgb <- end.time - start.time
print("time_xgb")
print(time_xgb)


###Artificial Neural Network###
start.time <- Sys.time()
model_ann <- train(as.factor(psych_out)~., data = train, 
                   method = "nnet",
                   preProcess = c("scale", "center"),
                   trControl = cv,
                   allowParallel = (n_cores > 1),
                   verbose = FALSE,
                   verbosity = 0)
test$ann_pred <- predict(model_ann, test, type="prob")
end.time <- Sys.time()
time_ann <- end.time - start.time
print("time_ann")
print(time_ann)


###Linear Discriminant Analysis###
start.time <- Sys.time()
model_lda <- train(as.factor(psych_out)~., data = train, 
                  method = "lda",
                  preProcess = c("scale", "center"),
                  allowParallel = (n_cores > 1),
                  trControl = cv,
                  verbose = FALSE,
                  verbosity = 0)
test$lda_pred <- predict(model_lda, test, type="prob")
end.time <- Sys.time()
time_lda <- end.time - start.time
print("time_lda")
print(time_lda)


if(n_cores > 1){
  stopCluster(cl)
}

save(test, file = paste0("output/", outco, "_test_predictions.RData"))
save(logit, file=paste0("output/", outco, "_logit.RData"))
save(model_en, file=paste0("output/", outco, "_elasticnet.RData"))
save(model_xgb, file=paste0("output/", outco, "_xgb.RData"))
save(model_ann, file=paste0("output/", outco, "_ann.RData"))
save(model_lda, file=paste0("output/", outco, "_lda.RData"))

###############################################
# Create the evaluation graph of the Test set #
###############################################
test$logit_risk_score_decile <- ntile(test$logit_pred, 10)  
test$en_risk_score_decile <- ntile(test$en_pred$case, 10)  
test$xgb_risk_score_decile <- ntile(test$xgb_pred$case, 10)  
test$lda_risk_score_decile <- ntile(test$lda_pred$case, 10)  
test$ann_risk_score_decile <- ntile(test$ann_pred$case, 10)  

Outcome_order <- c("Decile 2", "Decile 3", "Decile 4", "Decile 5", "Decile 6", "Decile 7",
                   "Decile 8", "Decile 9", "Decile 10")

variables <- c("logit","xgb", "en", "lda", "ann")
results_list <- list()
for (var in variables) {
  x_prev <- paste0(var, "_prev")
  x_decile <- paste0(var, "_risk_score_decile")
  x_output <- paste0(var, "_output")
  x_dec <- paste0(var, "_dec")
  
  deci_tmp <- glm(as.factor(test$psych_out) ~ as.factor(test[[x_decile]]), family = binomial(link="logit"))
  
  tmp_out <- as.data.frame(exp(deci_tmp$coefficients))
  tmp_out <- rename(tmp_out,OR=`exp(deci_tmp$coefficients)`)
  tmp_out$se <- summary(deci_tmp)$coefficients[,2]
  tmp_out <- tmp_out[-1,]
  tmp_out$Lower <- exp(log(tmp_out$OR)-1.96*tmp_out$se)
  tmp_out$Upper <- exp(log(tmp_out$OR)+1.96*tmp_out$se)
  tmp_out$Outcome <- Outcome_order
  tmp_out$Model <- var

  tmp_out$psych_out <- table(test[[x_decile]][test$psych_out == 1 & test[[x_decile]]!=1])
  tmp_out$total <- table(test[[x_decile]][test[[x_decile]]!=1])
  tmp_out$prev <- round(tmp_out$psych_out/tmp_out$total*100,2)
  tmp_out$text <- paste0(tmp_out$psych_out, " (",tmp_out$prev, "%)" )
  
  assign(x_output, tmp_out)
  print(paste0(var, " done"))
}

df = rbind(logit_output, en_output, xgb_output, lda_output, ann_output)
df$Outcome = factor (df$Outcome, level=Outcome_order)
df$Model <- ifelse(df$Model=="logit", "Logistic", df$Model)
df$Model <- ifelse(df$Model=="xgb", "XGB", df$Model)
df$Model <- ifelse(df$Model=="lda", "LDA", df$Model)
df$Model <- ifelse(df$Model=="en", "EN", df$Model)
df$Model <- ifelse(df$Model=="ann", "ANN", df$Model)

#define colours for dots and bars
dotCOLS = c("#E69F00","#56B4E9","#009E73",
            "#F0E442","#0072B2","#D55E00")

barCOLS = c("#CCC","#CCC", "#CCC","#CCC","#CCC","#CCC")

p_no_log <- ggplot(df, aes(x=Outcome, y=log(OR), ymin=log(Lower), ymax=log(Upper),col=Model,fill=Model)) + 
  geom_linerange(size=5,position=position_dodge(width = .9)) +
  geom_hline(yintercept=0, lty=2) +
  geom_point(size=3, shape=21, colour="white", stroke = 0.5,position=position_dodge(width = .9)) +
  scale_fill_manual(values=barCOLS)+
  scale_color_manual(values=dotCOLS)+
  scale_x_discrete(name="Risk decile") +
  scale_y_continuous(name="Beta", limits = c(-2, 5), breaks = c(-1,0, 1, 2,3, 4) ) +
  geom_text(aes(y=4.5, label=text, x=Outcome), color="black",position=position_dodge(width = 0.8)) +
  coord_flip() +
  theme_minimal() 

p_no_log 
ggsave(filename=paste0("output/", outco, "_models_estimation.jpg"), 
       plot=p_no_log, scale=1.5, 
       width = 7,
       height = 5
)


####################################################
### Save the model information from the test set ###
####################################################
test_tmp <- test
test_tmp$logit_pred <- test$logit_pred
test_tmp$en_pred <- test$en_pred$case
test_tmp$xgb_pred <- test$xgb_pred$case 
test_tmp$lda_pred <- test$lda_pred$case
test_tmp$ann_pred <- test$ann_pred$case

prediction_summary <- c("Model", "AUC", "Precision", "Recall", "F1")
for (var in variables) {
  x_pred <- paste0(var, "_pred")
  x_prediction_eval <- paste0(var, "_prediction_eval")
  response <- test_tmp[[x_pred]]
  auc <- auc(roc(test$psych_out, response))
  #Note that, depending on the response-probability threshold, Precision, Recall and F1 information can be very limited.
  #Since the ARI is a continuous score, such information is purely descriptive. 
  response <- ifelse(response>.5, 1,0)
    recall <- sensitivity(as.factor(response), as.factor(test$psych_out), positive="1")
  precision <- posPredValue(as.factor(response), as.factor(test$psych_out), positive="1")
  prediction_eval <-  c(var, 
                              as.character(round(auc,digits=3)),
                              as.character(round(precision,digits=5)),
                              as.character(round(recall,digits=5)),
                              as.character(round((2 * precision * recall) / (precision + recall),digits=5)))
  
  prediction_summary <- rbind(prediction_summary, prediction_eval)
  
}
times <- round(c(time_logit, time_xgb, time_en, time_lda, time_ann)/60,3)
times <- c("Minutes", as.character(times))
prediction_summary<- cbind(prediction_summary, times)
prediction_summary
write.csv(prediction_summary,paste0("output/", outco, "_evaluation_summary.csv"))

}







############################################################
######################## README ############################
############################################################

# This script generates out-of-sample ARI predictions for
# every individual in the dataset using 5-fold cross-validation.
#
# The purpose here is not model evaluation, but generation of
# prediction scores that can be used in downstream analyses.
#
# To avoid information leakage, each individual's prediction
# is generated by a model that was trained without using
# that individual's own data.
#
# This implementation uses XGBoost as an example, but the
# model-fitting section can easily be replaced by another
# machine-learning method.

############################################################

library(xgboost) # Caret can fail when the data are too large
                 # therefore we use an external ML package here. 
                 # It should yield largely the same estimates. 
library(Matrix)

############################################################
# Settings
############################################################

set.seed(seed)

############################################################
# Loop through outcomes
############################################################

for (outcome in names(datasets)) {
  cat("\n====================================\n")
  cat("Running outcome:", outcome, "\n")
  cat("====================================\n")
  data <- datasets[[outcome]]
  ##########################################################
  # Remove non-predictor variables
  ##########################################################
  ##########################################################
  # Remove non-predictor variables
  ##########################################################
  other_outcomes <- setdiff(names(datasets), outcome)
  vars_to_remove <- intersect(other_outcomes, names(data))
  vars_to_remove <- c(vars_to_remove, exclude_vars)
  data <- data[, !(names(data) %in% vars_to_remove)]
  
  ##########################################################
  # Ensure predictors are numeric
  ##########################################################
  
  predictor_vars <- setdiff(names(data),c(outcome, id_variable))
  data[predictor_vars] <- lapply(data[predictor_vars],as.numeric)
  
  ##########################################################
  # Create folds
  ##########################################################
  
  data$fold <- sample(rep(seq_len(n_folds),length.out = nrow(data)))
  folds <- split(data, data$fold)
  prediction_list <- vector("list",n_folds)
  
  ##########################################################
  # Cross-validation loop
  ##########################################################
  
  for (i in seq_len(n_folds)) {
    cat("Fold:", i, "\n")
    test <- folds[[i]]
    train <- do.call(rbind,folds[-i])
    
    ########################################################
    # Prepare XGBoost matrices
    ########################################################
    
    train[[outcome]] <- as.numeric(train[[outcome]] == 1)
    full <- rbind(train[, !(names(train) %in% c(outcome, id_variable, "fold"))],
                  test[, !(names(test) %in% c(outcome, id_variable, "fold"))])
    
    X_all <- sparse.model.matrix( ~ . - 1, data = full)
    X <- X_all[1:nrow(train),]
    
    X_test <- X_all[(nrow(train) + 1):nrow(X_all),]
    y <- train[[outcome]]
    dtrain <- xgb.DMatrix(X,label = y)
    
    ########################################################
    # XGBoost hyperparameter tuning
    ########################################################
    
    grid <- expand.grid(
      max_depth = c(3, 4, 5),
      eta = c(0.03, 0.05),
      min_child_weight = c(1, 5),
      subsample = 0.8,
      colsample_bytree = 0.8
    )
    
    best_logloss <- Inf
    
    for (k in seq_len(nrow(grid))) {
      
      params <- list(
        objective = "reg:logistic",
        eval_metric = "logloss",
        max_depth = grid$max_depth[k],
        eta = grid$eta[k],
        min_child_weight = grid$min_child_weight[k],
        subsample = grid$subsample[k],
        colsample_bytree = grid$colsample_bytree[k],
        lambda = 2,
        alpha = 0.5,
        nthread = max(1, n_cores))
      
      cv_model <- xgb.cv(
        data = dtrain,
        params = params,
        nfold = 5,
        nrounds = 1000,
        early_stopping_rounds = 30,
        verbose = 0
      )
      
      min_logloss <- min(cv_model$evaluation_log$test_logloss_mean)
      best_iter <- which.min(cv_model$evaluation_log$test_logloss_mean)
      
      if (min_logloss < best_logloss) {
        
        best_logloss <- min_logloss
        best_params <- params
        best_nrounds <- best_iter
        
      }
    }
    
    ########################################################
    # Final model
    ########################################################
    
    final_model <- xgb.train(
      data = dtrain,
      params = best_params,
      nrounds = best_nrounds
    )
    
    ########################################################
    # Prediction
    ########################################################
    
    pred_name <- paste0(outcome,"_prediction_ARI")
    test[[pred_name]] <- predict(final_model,X_test)
    prediction_list[[i]] <- test[, c(id_variable,
                                     birthyear_variable,
                                     pred_name)]
    gc()
  }
  
  ##########################################################
  # Combine predictions
  ##########################################################
  
  prediction_export <- do.call(rbind, prediction_list)
  pred_name <- paste0(outcome, "_prediction_ARI")
  ##########################################################
  # Standardize ARI within birth year
  ##########################################################
  
  prediction_export <- prediction_export %>%
    group_by(.data[[birthyear_variable]]) %>%
    mutate(
      ari_z = as.numeric(scale(.data[[pred_name]]))
    ) %>%
    ungroup()
  
  ##########################################################
  # Create overall deciles from standardized scores
  ##########################################################
  
  prediction_export$ari_decile <- ntile(
    prediction_export$ari_z,
    10
  )
  
  ##########################################################
  # Split top decile into 10 equal groups
  ##########################################################
  
  prediction_export$ari_group <-
    as.character(prediction_export$ari_decile)
  
  top_decile <- prediction_export$ari_decile == 10
  
  prediction_export$ari_group[top_decile] <-
    paste0(
      "10.",
      ntile(
        prediction_export$ari_z[top_decile],
        10
      )
    )  
  
  ##########################################################
  # Save output
  ##########################################################
  prediction_export[[birthyear_variable]] <- NULL
  names(prediction_export)[names(prediction_export) == "ari_z"] <-
    paste0(outcome, "_z_ARI")
  names(prediction_export)[names(prediction_export) == "ari_decile"] <-
    paste0(outcome, "_decile_ARI")
  names(prediction_export)[names(prediction_export) == "ari_group"] <-
    paste0(outcome, "_strata_ARI")
  
  write_dta(prediction_export,paste0("output/ari_",outcome,"_predictions.dta"))
  
  cat("\nSaved:",paste0("output/ari_",outcome,"_predictions.dta"),"\n")
}






