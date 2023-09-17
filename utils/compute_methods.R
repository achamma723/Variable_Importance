np <- import("numpy", convert = FALSE)
sklearn <- import("sklearn", convert = FALSE)
shap <- import("shap", convert = FALSE)
sandbox <- import("sandbox", convert = FALSE)
BBI <- import("BBI", convert = FALSE)


compute_d0crt <- function(sim_data,
                          seed,
                          loss = "least_square",
                          statistic = "residual",
                          ntree = 100L,
                          prob_type = "regression",
                          verbose = FALSE,
                          scaled_statistics = FALSE,
                          refit = FALSE,
                          ...) {
  print("Applying d0CRT Method")
  
  d0crt_results <- sandbox$dcrt_zero(
    sim_data[, -1],
    as.numeric(sim_data$y),
    loss = loss,
    screening = FALSE,
    statistic = statistic,
    ntree = ntree,
    type_prob = prob_type,
    refit = refit,
    scaled_statistics = scaled_statistics,
    verbose = TRUE,
    random_state = seed
  )

  return(data.frame(
    method = ifelse(scaled_statistics,
                    "d0CRT_scaled",
                    "d0CRT"
    ),
    importance = as.vector(d0crt_results[[2]]),
    p_value = as.vector(d0crt_results[[1]]),
    score = as.character(d0crt_results[[3]])
  ))
}


compute_strobl <- function(sim_data,
                           ntree = 100L,
                           mtry = 5L,
                           conditional = TRUE,
                           prob_type = "regression",
                           ...) {
  print("Applying Strobl Method")

  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  sim_data$y <- as.numeric(sim_data$y)
  
  f1 <- cforest(y ~ .,
                data = sim_data[train_ind, ],
                control = cforest_unbiased(ntree = ntree,
                                           mtry = mtry
                )
  )
  
  result <- permimp(f1,
                    conditional = conditional,
                    nperm = 100L,
                    progressBar = FALSE
  )
  
  if (prob_type == "classification") {
    pred <- predict(f1, newdata = sim_data[-train_ind, -1], type = "response")
    score <- sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred)
  }
  else {
    pred <- predict(f1, newdata = sim_data[-train_ind, -1])
    score <- sklearn$metrics$r2_score(sim_data$y[-train_ind], pred)
  }

  return(data.frame(
    method = "Strobl",
    importance = as.numeric(result$values),
    p_value = ifelse(is.nan(result$p_val), 1.0, result$p_val),
    score = py_to_r(score)
  ))
}


compute_shap <- function(sim_data,
                         seed = 2021L,
                         ntree = 100L,
                         prob_type = "regression",
                         ...) {
  print("Applying SHAP Method")
  
  if (prob_type == "classification") {
    clf_rf <- sklearn$ensemble$
      RandomForestClassifier(n_estimators = ntree)
  }
  
  if (prob_type == "regression") {
    clf_rf <- sklearn$ensemble$
      RandomForestRegressor(n_estimators = ntree)
  }
  
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
  explainer <- shap$TreeExplainer(clf_rf)
  
  if (prob_type == "classification") {
    shap_values <- as.matrix(explainer$shap_values(sim_data[-train_ind, -1])[[1]])
  }
  if (prob_type == "regression") {
    shap_values <- as.matrix(explainer$shap_values(sim_data[-train_ind, -1]))
  }
  
  return(data.frame(
    method = "Shap",
    importance = colMeans(shap_values),
    p_value = NA,
    score = NA
  ))
}


compute_sage <- function(sim_data,
                         seed = 2021L,
                         ntree = 100L,
                         prob_type = "regression",
                         ...) {
  print("Applying SAGE Method")
  utilspy <- import_from_path("utils_py",
                              path = "utils"
  )
  imp <- utilspy$compute_sage(
    sim_data[, -1],
    sim_data$y,
    ntree,
    seed,
    prob_type
  )
  
  return(data.frame(
    method = "SAGE",
    importance = imp,
    p_value = NA,
    score = NA
  ))
}


compute_mdi <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
  print("Applying MDI Method")
  
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)

  if (prob_type == "classification") {
    sim_data$y <- as.factor(sim_data$y)
    clf_rf <- sklearn$ensemble$
      RandomForestClassifier(n_estimators = ntree)
    clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
    pred <- py_to_r(clf_rf$predict_proba(sim_data[-train_ind, -1]))[, 2]
    score <- sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred)
  }
  
  if (prob_type == "regression") {
    sim_data$y <- as.numeric(sim_data$y)
    clf_rf <- sklearn$ensemble$
      RandomForestRegressor(n_estimators = ntree)
    clf_rf$fit(sim_data[train_ind, -1], sim_data$y[train_ind])
    pred <- clf_rf$predict(sim_data[-train_ind, -1])
    score <- sklearn$metrics$r2_score(sim_data$y[-train_ind], pred)
  }
  
  return(data.frame(
    method = "MDI",
    importance = as.numeric(clf_rf$feature_importances_),
    p_value = NA,
    score = py_to_r(score))
  )
}


compute_marginal <- function(sim_data,
                             prob_type = "regression",
                             list_grps = list(),
                             ...) {
  print("Applying Marginal Method")
  
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  marginal_imp <- numeric()
  marginal_pval <- numeric()
  score_val <- 0
  if (length(list_grps) == 0)
    indices = paste0("x", 1:ncol(sim_data[, -1]))
  else
    indices = list_grps
  
  count_ind = 1
  if (prob_type == "classification") {
    sim_data$y <- as.factor(sim_data$y)
    for (i in indices) {
      i = paste0(i, collapse="+")
      fit <- glm(formula(paste0("y ~ ", i)),
                 data = sim_data[train_ind, ],
                 family = binomial()
      )
      sum_fit <- summary(fit)
      marginal_imp[count_ind] <- coef(sum_fit)[, 1][[2]]
      marginal_pval[count_ind] <- coef(sum_fit)[, 4][[2]]
      pred <- predict(fit, newdata = sim_data[-train_ind, -1], type="response")
      score_val <- score_val +
        py_to_r(sklearn$metrics$roc_auc_score(sim_data$y[-train_ind], pred))
      count_ind <- count_ind + 1
    }
  }
  
  if (prob_type == "regression") {
    sim_data$y <- as.numeric(sim_data$y)
    for (i in indices) {
      i = paste0(i, collapse="+")
      fit <- glm(formula(paste0("y ~ ", i)),
                 data = sim_data[train_ind, ]
      )
      sum_fit <- summary(fit)
      marginal_imp[count_ind] <- coef(sum_fit)[, 1][[2]]
      marginal_pval[count_ind] <- coef(sum_fit)[, 4][[2]]
      pred <- predict(fit, newdata = sim_data[-train_ind, -1])
      score_val <- score_val + py_to_r(sklearn$metrics$r2_score(sim_data$y[-train_ind], pred))
      count_ind <- count_ind + 1
    }
  }
  
  return(data.frame(
    method = "Marg",
    importance = marginal_imp,
    p_value = marginal_pval,
    score = score_val / ncol(sim_data[, -1])
  ))
}


compute_bart <- function(sim_data,
                         ntree = 100L,
                         num_cores = 4,
                         prob_type = "regression",
                         ...) {
  print("Applying BART Method")
  
  if (prob_type == "classification") {
    sim_data$y <- as.factor(sim_data$y)
    score_fn <- sklearn$metrics$roc_auc_score
  }
  if (prob_type == "regression") {
    sim_data$y <- as.numeric(sim_data$y)
    score_fn <- sklearn$metrics$r2_score
  }
  options(java.parameters = "-Xmx10000m")
  library(bartMachine)
  
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  
  bart_machine <- bartMachine(
    X = sim_data[train_ind, -1],
    y = sim_data$y[train_ind],
    num_trees = ntree,
    mem_cache_for_speed = FALSE,
    verbose = FALSE
  )
  
  imp <- investigate_var_importance(bart_machine,
                                    plot = FALSE    
  )$avg_var_props
  imp <- imp[mixedsort(names(imp))]
  
  pred = bart_predict_for_test_data(bart_machine,
                                    sim_data[-train_ind, -1],
                                    sim_data$y[-train_ind])
  
  if (prob_type == "classification") {
    pred = 1 - pred$p_hat
  }
  if (prob_type == "regression") {
    pred = pred$y_hat
  }
  return(data.frame(
    method = "BART",
    importance = as.numeric(imp),
    p_value = NA,
    score = as.character(score_fn(sim_data$y[-train_ind], pred))
  ))
}


compute_knockoff <- function(sim_data,
                             seed,
                             save_file,
                             stat_knockoff = NULL,
                             with_bart = TRUE,
                             verbose = TRUE,
                             prob_type = "regression",
                             ...) {
  sim_data$y <- as.numeric(sim_data$y)
  if (prob_type == "classification") {
    if (stat_knockoff == "lasso_cv") {
      stat_knockoff = "logistic_l1"
    }
    else
      sim_data$y <- as.factor(sim_data$y)
  }
  
  res <- sandbox$model_x_knockoff(sim_data[, -1],
                                  sim_data$y,
                                  statistics = stat_knockoff,
                                  verbose = verbose,
                                  save_file = save_file,
                                  prob_type = prob_type,
                                  seed = seed
  )
  
  if (stat_knockoff == "l1_regu_path") {
    return(data.frame(
      method = "Knockoff_path",
      importance = res[[2]][1:as.integer(length(res[[2]]) / 2)],
      p_value = NA,
      score = NA
    ))
  } else if (stat_knockoff == "bart") {
    print("Applying Knockoff-BART Method")
    res_imp <- compute_bart(data.frame(y = sim_data$y,
                                       as.matrix(res[[0]])),
                            prob_type = prob_type)
    test_score <- res_imp$importance[1:ncol(sim_data[, -1])]
    -res_imp$importance[ncol(sim_data[, -1]):(2 * ncol(sim_data[, -1]))]
    
    return(data.frame(
      method = "Knockoff_bart",
      importance = as.vector(test_score),
      p_value = NA,
      score = as.character(res_imp$score[1])
    ))
  } else if (stat_knockoff == "deep") {

    print("Applying Knockoff-Deep Method")
    return(data.frame(
      method = "Knockoff_deep",
      importance = as.vector(res[0]),
      p_value = NA,
      score = as.character(res[1])
    ))
  }
  print("Applying Knockoff-Lasso Method")
  return(data.frame(
    method = "Knockoff_lasso",
    importance = as.vector(res[1]),
    p_value = NA,
    score = as.character(res[2])
  ))
}


compute_permfit <- function(sim_data,
                    index_i,
                    n = 1000L,
                    prob_type = "regression",
                    n_perm = 100,
                    n_jobs = 1,
                    nominal = NULL,
                    ...) {
  print("Applying Permfit-DNN Method")
  
  bbi_model <- BBI$BlockBasedImportance(
        prob_type = prob_type,
        index_i = index_i,
        conditional = FALSE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal)

  bbi_model$fit(
      sim_data[, -1],
      as.matrix(sim_data$y)
  )

  results <- bbi_model$compute_importance()
    if(prob_type == "regression")
        score = as.character(results$score_R2)
    else
        score = as.character(results$score_AUC)

  return(data.frame(
        method = "Permfit-DNN",
        importance = as.matrix(results$importance)[, 1],
        p_value = as.matrix(results$pval)[, 1],
        score = score
  ))
}


compute_cpi <- function(sim_data,
                index_i,
                n = 1000L,
                prob_type = "regression",
                n_perm = 100,
                n_jobs = 1,
                nominal = NULL,
                ...) {
  print("Applying CPI-DNN Method")
  
  bbi_model <- BBI$BlockBasedImportance(
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal)

  bbi_model$fit(
      sim_data[, -1],
      as.matrix(sim_data$y)
  )

  results <- bbi_model$compute_importance()
    if(prob_type == "regression")
        score = as.character(results$score_R2)
    else
        score = as.character(results$score_AUC)

  return(data.frame(
        method = "CPI-DNN",
        importance = as.matrix(results$importance)[, 1],
        p_value = as.matrix(results$pval)[, 1],
        score = score
  ))
}


compute_cpi_rf <- function(sim_data,
                   index_i,
                   n = 1000L,
                   prob_type = "regression",
                   n_perm = 100,
                   n_jobs = 1,
                   nominal = NULL,
                   ...) {
  print("Applying CPI-RF Method")
  
  bbi_model <- BBI$BlockBasedImportance(
        estimator = "RF",
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        n_perm = n_perm,
        n_jobs = n_jobs,
        list_nominal = nominal)

  bbi_model$fit(
      sim_data[, -1],
      as.matrix(sim_data$y)
  )

  results <- bbi_model$compute_importance()
    if(prob_type == "regression")
        score = as.character(results$score_R2)
    else
        score = as.character(results$score_AUC)

  return(data.frame(
        method = "CPI-RF",
        importance = as.matrix(results$importance)[, 1],
        p_value = as.matrix(results$pval)[, 1],
        score = score
  ))
}


compute_lazy <- function(sim_data,
                         prob_type = "regression",
                         ...) {
  print("Applying Lazy Method")
  utilspy <- import_from_path("utils_py",
                              path = "utils"
  )
  imp <- utilspy$compute_lazy(
    as.matrix(sim_data[, -1]),
    np$array(as.numeric(sim_data$y))
  )
  
  p_value <- c()
  for(i in 1:length(imp$ub_list)){
    if ((0 <= imp$ub_list[[i]]) & (0 >= imp$lb_list[[i]]))
      p_value <- c(p_value, 0)
    else
      p_value <- c(p_value, 1)
  }
  
  return(data.frame(
    method = 'lazyvi',
    importance = as.vector(imp$imp_vals),
    p_value = p_value,
    score = NA
  ))
}

compute_cpi_knockoff <- function(sim_data,
                        prob_type = "regression",
                        ...) {
  print("Applying CPI-Knockoff Method")

  if(prob_type == "classification")
    sim_data$y <- as.factor(sim_data$y)
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  train_data <- sim_data[train_ind, ]
  test_data <- sim_data[-train_ind, ]
  
  if(prob_type=='regression'){
    mytask <- as_task_regr(train_data, target='y')
    mylearner <- lrn("regr.ranger", predict_type = "response", keep.inbag = TRUE, num.trees=500)
    res <- cpi(task = mytask, 
               learner = mylearner,
               test_data = test_data,
               measure = "regr.mse")
  }
  else{
    mytask <- as_task_classif(x=train_data, target='y')
    mylearner <- lrn("classif.ranger", predict_type = "prob", keep.inbag = TRUE, num.trees=500)
    res <- cpi(task = mytask, 
               learner = mylearner,
               test_data = test_data,
               measure = "classif.logloss")
  }
  
  return(data.frame(
    method = 'cpi_knockoff',
    importance = as.vector(res$estimate),
    p_value = as.vector(res$p.value),
    score = NA
  ))
}

compute_loco <- function(sim_data,
                         dnn = FALSE,
                         seed = 2021L,
                         ntree = 100L,
                         prob_type = "regression",
                         ...) {
  print("Applying LOCO Method")
  utilspy <- import_from_path("utils_py",
                              path = "utils"
  )
  imp <- utilspy$compute_loco(
    sim_data[, -1],
    as.vector(sim_data$y),
    ntree,
    seed,
    prob_type,
    dnn=dnn
  )
  if (dnn==FALSE){
    method <- 'loco'
  }else{
    method <- 'LOCO-DNN'
  }

  return(data.frame(
    method = method,
    importance = unlist(imp$val_imp),
    p_value = unlist(imp$p_value),
    score = NA
  ))
}