############################################################
# ARI_functions.R
#
# Copyright (c) 2026 Hugo Sjöqvist
#
# Released under the MIT License.
#
# This code relies on third-party R packages,
# including xgboost, caret, Matrix, dplyr,
# foreach and doParallel, which are subject
# to their own licenses.
############################################################

##########################################################
##########################################################
##########################################################
############## ARI model evaluation ######################
##########################################################
##########################################################
##########################################################
##########################################################


ari_model_evaluation <- function(
    data,
    outcome,
    id = NULL,
    exclude_vars = NULL,
    seed = 242323,
    train_fraction = 0.8,
    n_cores = 1
) {
  
  ##########################################################
  # Input checks
  ##########################################################
  
  if (!dir.exists("output")) {
    dir.create("output")
  }
  if (!dir.exists("data")) {
    stop("A 'data' folder containing input datasets is required.")
  }
  
  if (!outcome %in% names(data)) {
    stop("Outcome variable not found in data.")
  }
  
  if (!all(data[[outcome]] %in% c(0, 1))) {
    stop("Outcome must be coded as 0/1.")
  }
  
  if (!is.character(outcome) || length(outcome) != 1) {
    stop("Outcome must be provided as a single column name, e.g. outcome='asd'.")
  }
  
  if (!is.null(id) && (!is.character(id) || length(id) != 1)) {
    stop("id must be provided as a column name, e.g. id='id'.")
  }
  
  ##########################################################
  # Data preparation
  ##########################################################
  
  set.seed(seed)
  
  #Removing the ID and unwanted variables for the evaluation.
  remove_vars <- c(id, exclude_vars)  
  remove_vars <- remove_vars[remove_vars %in% names(data)]
  data <- data[, !(names(data) %in% remove_vars)]
  
  data$psych_out <- data[[outcome]]
  data[[outcome]] <- NULL
  
  predictor_vars <- setdiff(names(data), "psych_out")
    data[predictor_vars] <- lapply(data[predictor_vars],as.numeric)
  
  ##########################################################
  # Train/test split
  ##########################################################
  
  samp <- sample(
    nrow(data),
    floor(train_fraction * nrow(data)))
  
  train <- data[samp, ]
  test  <- data[-samp, ]
  

  ##########################################################
  # Benchmark logistic regression
  ##########################################################
  
  start_logit <- Sys.time()
  
  model_logit <- glm(
    psych_out ~ .,
    data = train,
    family = binomial()
  )
  
  test$logit_pred <- predict(
    model_logit,
    test,
    type = "response"
  )
  
  time_logit <- Sys.time() - start_logit
  
  ##########################################################
  # XGBoost ARI model
  ##########################################################
  
  train$psych_out <- factor(
    ifelse(train$psych_out == 1,
           "case",
           "control")
  )

  if (n_cores > 1) {
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)
  }
  
  cv <- caret::trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE
  )

  start_xgb <- Sys.time()
  
  model_xgb <- caret::train(
    as.factor(psych_out) ~ .,
    data = train,
    method = "xgbTree",
    preProcess = c("center", "scale"),
    trControl = cv,
    allowParallel = (n_cores > 1) ,
    verbose = FALSE
  )
  
  
  test$xgb_pred <- predict(
    model_xgb,
    test,
    type = "prob"
  )$case
  
  time_xgb <- Sys.time() - start_xgb
  

  ##########################################################
  # Evaluation metrics
  ##########################################################
  
  get_metrics <- function(obs, pred) {
    auc_val <- as.numeric(pROC::auc(pROC::roc(obs, pred)))
    data.frame(AUC = round(auc_val, 3))
  }
  
  summary_table <- rbind(
    cbind(
      Model = "Logistic",
      get_metrics(
        test$psych_out,
        test$logit_pred
      ),
      Runtime_Minutes =
        round(as.numeric(time_logit, units = "mins"), 2)
    ),
    
    cbind(
      Model = "ARI_XGB",
      get_metrics(
        test$psych_out,
        test$xgb_pred
      ),
      Runtime_Minutes =
        round(as.numeric(time_xgb, units = "mins"), 2)
    )
    
  )

  ##########################################################
  # Return results
  ##########################################################
  
  return(
    list(
      summary = summary_table,
      predictions = test,
      logistic_model = model_logit,
      ari_model = model_xgb
    )
  )
  
}

  

  
  

  ##########################################################
  ##########################################################
  ##########################################################
  ############## ARI model prediction ######################
  ##########################################################
  ##########################################################
  ##########################################################
  ##########################################################

ari_prediction <- function(
    data,
    outcome,
    id,
    birthyear,
    exclude_vars = NULL,
    seed = 242323,
    n_folds = 5,
    n_cores = 1
) {
  
  ##########################################################
  # Checks
  ##########################################################
  
  if (!(outcome %in% names(data))) {
    stop("Outcome variable not found.")
  }
  
  if (!(id %in% names(data))) {
    stop("ID variable not found.")
  }
  
  if (!(birthyear %in% names(data))) {
    stop("Birthyear variable not found.")
  }
  
  if (!all(data[[outcome]] %in% c(0, 1))) {
    stop("Outcome must be coded as 0/1.")
  }
  
  set.seed(seed)
  
  ##########################################################
  # Remove unwanted variables
  ##########################################################
  
  vars_to_remove <- exclude_vars
  vars_to_remove <- vars_to_remove[
    vars_to_remove %in% names(data)
  ]
  
  data <- data[
    ,
    !(names(data) %in% vars_to_remove)
  ]
  
  ##########################################################
  # Convert predictors to numeric
  ##########################################################
  
  predictor_vars <- setdiff(
    names(data),
    c(outcome, id)
  )
  
  data[predictor_vars] <- lapply(
    data[predictor_vars],
    as.numeric
  )
  
  ##########################################################
  # Create folds
  ##########################################################
  
  data$fold <- sample(
    rep(
      seq_len(n_folds),
      length.out = nrow(data)
    )
  )
  
  folds <- split(
    data,
    data$fold
  )
  
  ##########################################################
  # Parallel setup
  ##########################################################
  
  if (n_cores > 1) {
    
    cl <- parallel::makeCluster(n_cores)
    
    doParallel::registerDoParallel(cl)
    
    on.exit(
      try(
        parallel::stopCluster(cl),
        silent = TRUE
      ),
      add = TRUE
    )
    
  } else {
    
    foreach::registerDoSEQ()
    
  }
  
  ##########################################################
  # Cross-validation prediction
  ##########################################################
  message(
    "Starting ARI prediction using ",
    n_folds,
    " folds and ",n_cores," core(s)...")
  start_time <- Sys.time()
  start_time <- Sys.time()
  prediction_list <- foreach::foreach(
    i = seq_len(n_folds),
    .packages = c(
      "xgboost",
      "Matrix",
      "dplyr"
    )
  ) %dopar% {
    
    test <- folds[[i]]
    
    train <- do.call(
      rbind,
      folds[-i]
    )
    
    ########################################################
    # Create sparse matrices
    ########################################################
    
    y <- train[[outcome]]
    
    full <- rbind(
      train[
        ,
        !(names(train) %in%
            c(outcome, id, birthyear, "fold"))
      ],
      test[
        ,
        !(names(test) %in%
            c(outcome, id, birthyear, "fold"))
      ]
    )
    
    X_all <- Matrix::sparse.model.matrix(
      ~ . - 1,
      data = full
    )
    
    X_train <- X_all[
      1:nrow(train),
    ]
    
    X_test <- X_all[
      (nrow(train) + 1):nrow(X_all),
    ]
    
    dtrain <- xgboost::xgb.DMatrix(
      data = X_train,
      label = y
    )
    
    ########################################################
    # Hyperparameter grid search
    ########################################################
    
    grid <- expand.grid(
      max_depth = c(3, 4, 5),
      eta = c(0.03, 0.05),
      min_child_weight = c(1, 5),
      subsample = 0.8,
      colsample_bytree = 0.8
    )
    
    best_logloss <- Inf
    best_params <- NULL
    best_nrounds <- NULL
    
    for (k in seq_len(nrow(grid))) {
      
      params <- list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = grid$max_depth[k],
        eta = grid$eta[k],
        min_child_weight =
          grid$min_child_weight[k],
        subsample =
          grid$subsample[k],
        colsample_bytree =
          grid$colsample_bytree[k],
        lambda = 2,
        alpha = 0.5,
        nthread = 1
      )
      
      cv_model <- xgboost::xgb.cv(
        params = params,
        data = dtrain,
        nfold = 5,
        nrounds = 1000,
        early_stopping_rounds = 30,
        verbose = 0
      )
      
      min_logloss <- min(
        cv_model$evaluation_log$
          test_logloss_mean
      )
      
      best_iter <- which.min(
        cv_model$evaluation_log$
          test_logloss_mean
      )
      
      if (min_logloss < best_logloss) {
        
        best_logloss <- min_logloss
        best_params <- params
        best_nrounds <- best_iter
        
      }
      
    }
    
    ########################################################
    # Train final model
    ########################################################
    
    final_model <- xgboost::xgb.train(
      params = best_params,
      data = dtrain,
      nrounds = best_nrounds,
      verbose = 0
    )
    
    ########################################################
    # Predict held-out fold
    ########################################################
    
    fold_pred <- predict(
      final_model,
      X_test
    )
    
    data.frame(
      participant_id = test[[id]],
      birthyear = test[[birthyear]],
      ari_prediction = fold_pred
    )
    

  }
  
  ##########################################################
  # Combine predictions
  ##########################################################
  
  prediction_export <- do.call(
    rbind,
    prediction_list
  )
  message(
    "ARI prediction completed in ",
    round(
      as.numeric(
        difftime(
          Sys.time(),
          start_time,
          units = "mins"
        )
      ),
      2
    ),
    " minutes."
  )
  
  ##########################################################
  # Birthyear-specific z-score
  ##########################################################
  
  prediction_export$ari_z <- ave(
    prediction_export$ari_prediction,
    prediction_export$birthyear,
    FUN = function(x)
      as.numeric(scale(x))
  )
  
  ##########################################################
  # Deciles based on z-score
  ##########################################################
  
  prediction_export$ari_decile <- dplyr::ntile(
    prediction_export$ari_z,
    10
  )
  
  ##########################################################
  # Create 10.1-10.10 strata
  ##########################################################
  
  prediction_export$ari_strata <-
    as.character(
      prediction_export$ari_decile
    )
  
  top_decile <- prediction_export$ari_decile == 10
  
  prediction_export$top_subgroup <- NA
  
  prediction_export$top_subgroup[top_decile] <-
    dplyr::ntile(
      prediction_export$ari_z[top_decile],
      10
    )
  
  prediction_export$ari_strata[top_decile] <-
    paste0(
      "10.",
      prediction_export$top_subgroup[top_decile]
    )
  
  prediction_export$top_subgroup <- NULL
  
  ##########################################################
  # Make ordered factor
  ##########################################################
  
  prediction_export$ari_strata <- factor(
    prediction_export$ari_strata,
    levels = c(
      as.character(1:9),
      paste0("10.", 1:10)
    )
  )
  
  ##########################################################
  # Sort by ID
  ##########################################################
  
  prediction_export <-
    prediction_export[
      order(
        prediction_export$participant_id
      ),
    ]
  
  ##########################################################
  # Rename ID column
  ##########################################################
  
  names(prediction_export)[
    names(prediction_export) ==
      "participant_id"
  ] <- id
  
  ##########################################################
  # Return output
  ##########################################################
  
  prediction_export <-
    prediction_export[
      ,
      c(
        id,
        "ari_prediction",
        "ari_z",
        "ari_decile",
        "ari_strata"
      )
    ]
  
  return(prediction_export)
  
}
