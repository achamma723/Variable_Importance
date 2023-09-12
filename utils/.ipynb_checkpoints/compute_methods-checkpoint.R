# np <- import("numpy", convert = FALSE)
# alib <- import("alibi.explainers", convert = FALSE)
# sklearn <- import("sklearn", convert = FALSE)
# shap <- import("shap", convert = FALSE)
# sandbox <- import_from_path("sandbox",
#     path = "../tuan_binh_nguyen/dev"
# )

# source("utils/gimp.R")

compute_janitza <- function(sim_data,
                            cv,
                            ntree = 5000L,
                            ncores = 4L,
                            with_ranger = FALSE,
                            with_impurity = FALSE,
                            with_python = FALSE,
                            replace = FALSE,
                            prob_sim_data = "regression",
                            ...) {
  print("Applying HoldOut Method")
  prob_type <- strsplit(as.character(prob_sim_data), "_")[[1]][1]
  
  mtry <- ceiling(sqrt(ncol(sim_data[, -1])))
  
  if (prob_type == "classification") {
    sim_data$y <- as.factor(sim_data$y)
  }
  
  if (!with_ranger) {
    res <- CVPVI(sim_data[, -1],
                 sim_data$y,
                 k = cv,
                 mtry = mtry,
                 ntree = ntree,
                 ncores = ncores
    )
    tryCatch(
      {
        return(data.frame(
          method = "HoldOut_nr",
          importance = as.numeric(res$cv_varim),
          p_value = as.numeric(NTA(res$cv_varim)$pvalue)
        ))
      },
      finally = {
        return(data.frame(
          method = "HoldOut_nr",
          importance = as.numeric(res$cv_varim)
        ))
      }
    )
  }
  
  else {
    if (!with_impurity) {
      rf_sim <- holdoutRF(y ~ .,
                          data = sim_data,
                          mtry = mtry,
                          num.trees = ntree
      )
      suffix <- "r"
    }
    else {
      rf_sim <- ranger(y ~ .,
                       data = sim_data,
                       importance = "impurity_corrected",
                       mtry = mtry,
                       replace = replace,
                       num.trees = ntree
      )
      suffix <- "ri"
    }
    
    res <- importance_pvalues(rf_sim, method = "janitza")
    return(data.frame(
      method = paste0("HoldOut_", suffix),
      importance = as.numeric(res[, 1]),
      p_value = as.numeric(res[, 2])
    ))
  }
}


compute_altmann <- function(sim_data,
                            nper = 100L,
                            ntree = 500L,
                            replace = FALSE,
                            prob_sim_data = "regression",
                            ...) {
  print("Applying Altmann Method")
  prob_type <- strsplit(as.character(prob_sim_data), "_")[[1]][1]
  
  if (prob_type == "classification") {
    sim_data$y <- as.factor(sim_data$y)
  }
  
  rf_altmann <- ranger(y ~ .,
                       data = sim_data,
                       importance = "permutation",
                       mtry = ceiling(sqrt(ncol(sim_data[, -1]))),
                       num.trees = ntree,
                       replace = replace
  )
  
  res <- data.frame(importance_pvalues(rf_altmann,
                                       method = "altmann",
                                       num.permutations = nper,
                                       formula = y ~ .,
                                       data = sim_data
  ))
  return(data.frame(
    method = "Altmann",
    importance = res[, 1],
    p_value = res[, 2]
  ))
}


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
    importance = d0crt_results[[3]],
    p_value = d0crt_results[[2]],
    score = d0crt_results[[4]]
  ))
}


compute_strobl <- function(sim_data,
                           ntree = 100L,
                           mtry = 5L,
                           conditional = TRUE,
                           prob_type = "regression",
                           ...) {
  print("Applying Strobl Method")
  library('party')
  library(permimp)
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  
  # if (prob_type == "classification") {
  #     sim_data$y <- as.factor(sim_data$y)
  # }
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
  # print(as.numeric(result$values))
  # print(as.numeric(result$p_val))
  # stop()
  
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


# compute_group_lasso <- function(sim_data,
#                                 seed = 2021L,
#                                 prob_type = "regression",
#                                 ...) {
#     print("Applying Group Lasso Method")
#     utilspy <- import_from_path("utils_py",
#         path = "utils"
#     )
#     imp <- utilspy$compute_group_lasso(
#         sim_data[, -1],
#         sim_data$y,
#         seed,
#         prob_type
#         )

#     return(data.frame(
#         method = "SAGE",
#         importance = imp,
#         p_value = NA,
#         score = NA
#     ))
# }


compute_mdi <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
  print("Applying MDI Method")
  
  # Splitting train/test sets
  train_ind <- sample(length(sim_data$y), size = length(sim_data$y) * 0.8)
  print(train_ind)
  stop()
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
  
  # Compute p-values with permutation approach
  res = sklearn$inspection$permutation_importance(clf_rf, sim_data[-train_ind, -1],
                                                  sim_data$y[-train_ind], n_repeats=100L)
  imp_mean = py_to_r(res$importances_mean)
  imp_std = py_to_r(res$importances_std)
  z_test = imp_mean / imp_std
  p_val = 1 - stats::pnorm(z_test)
  
  return(data.frame(
    method = "MDI",
    importance = as.numeric(clf_rf$feature_importances_),
    p_value = ifelse(is.nan(p_val), 1.0, p_val),
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
    # depth = NA
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
  
  p_val <- c()
  for(i in colnames(sim_data[, -1]))
    p_val <- c(p_val, cov_importance_test(bart_machine,
                                          covariates = i,
                                          plot = FALSE)$pval)
  
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
    p_value = p_val,
    score = py_to_r(score_fn(sim_data$y[-train_ind], pred))
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
  print("Applying Knockoff Method")
  
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
    res_imp <- compute_bart(data.frame(y = sim_data$y,
                                       res[[1]]),
                            prob_type = prob_type)
    test_score <- res_imp$importance[1:ncol(sim_data[, -1])]
    -res_imp$importance[ncol(sim_data[, -1]):(2 * ncol(sim_data[, -1]))]
    
    return(data.frame(
      method = "Knockoff_bart",
      importance = test_score,
      p_value = NA,
      score = res_imp$score[1]
    ))
  } else if (stat_knockoff == "deep") {
    return(data.frame(
      method = "Knockoff_deep",
      importance = res[[1]],
      p_value = NA,
      score = res[[2]]
    ))
  }
  return(data.frame(
    method = "Knockoff_lasso",
    importance = res[[2]],
    p_value = NA,
    score = res[[3]]
  ))
}


compute_ale <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
  print("Applying ALE Method")
  
  if (prob_type == "classification") {
    clf_rf <- sklearn$ensemble$
      RandomForestClassifier(n_estimators = ntree)
    clf_rf$fit(sim_data[, -1], sim_data$y)
    rf_ale <- alib$ALE(clf_rf$predict_proba)
  }
  if (prob_type == "regression") {
    clf_rf <- sklearn$ensemble$
      RandomForestRegressor(n_estimators = ntree)
    clf_rf$fit(sim_data[, -1], sim_data$y)
    rf_ale <- alib$ALE(clf_rf$predict)
  }
  rf_explain <- rf_ale$explain(as.matrix(sim_data[, -1]))
  imp <- c()
  for (i in 1:dim(sim_data[, -1])[[2]]) {
    imp <- c(
      imp,
      mean(as.vector(rf_explain$ale_values[[i - 1]]))
    )
  }
  return(data.frame(
    method = "Ale",
    importance = imp
  ))
}


compute_dnn <- function(sim_data,
                        n = 1000L,
                        ...) {
  print("Applying DNN Method")
  set.seed(NULL)
  ## 1.0 Hyper-parameters
  esCtrl <- list(
    n.hidden = c(50L, 40L, 30L, 20L),
    activate = "relu",
    l1.reg = 10**-4,
    early.stop.det = 1000L,
    n.batch = 50L,
    n.epoch = 200L,
    learning.rate.adaptive = "adam",
    plot = FALSE
  )
  n_ensemble <- 10L
  n_perm <- 100L
  dnn_obj <- importDnnet(
    x = sim_data[, -1],
    y = as.numeric(sim_data$y)
  )
  
  # PermFIT-DNN
  shuffle <- sample(n)
  
  dat_spl <- splitDnnet(dnn_obj, 0.8)
  permfit_dnn <- permfit(
    train = dat_spl$train,
    validate = dat_spl$valid,
    k_fold = 0,
    pathway_list = list(),
    n_perm = n_perm,
    method = "ensemble_dnnet",
    shuffle = shuffle,
    n.ensemble = n_ensemble,
    esCtrl = esCtrl
  )
  
  return(data.frame(
    method = "Permfit-DNN_old",
    importance = permfit_dnn@importance$importance,
    p_value = permfit_dnn@importance$importance_pval
  ))
}


compute_grf <- function(sim_data,
                        type_forest = "regression",
                        ...) {
  print("Applying GRF Method")
  
  if (type_forest == "regression") {
    forest <- regression_forest(sim_data[, -1],
                                as.numeric(sim_data$y),
                                tune.parameters = "all"
    )
  }
  if (type_forest == "quantile") {
    forest <- quantile_forest(sim_data[, -1],
                              as.numeric(sim_data$y),
                              quantiles = c(0.1, 0.3, 0.5, 0.7, 0.9)
    )
  }
  return(data.frame(
    method = paste0("GRF_", type_forest),
    importance = variable_importance(forest)[, 1]
  ))
}


compute_bart_py <- function(sim_data,
                            ntree = 100L,
                            ...) {
  print("Applying BART Python Method")
  utilspy <- import_from_path("utils_py",
                              path = "utils"
  )
  imp <- utilspy$compute_bart_py(
    sim_data[, -1],
    np$array(as.numeric(sim_data$y))
  )
  
  return(data.frame(
    method = "Bart_py",
    importance = as.numeric(imp)
  ))
}


compute_dnn_py <- function(sim_data,
                           index_i,
                           n = 1000L,
                           prob_type = "regression",
                           n_perm = 100,
                           n_jobs = 1,
                           backend = "loky",
                           nominal = NULL,
                           list_grps = list(),
                           group_stack = FALSE,
                           ...) {
  print("Applying DNN Permfit Method")
  
  deep_py <- import_from_path("permfit_py",
        path = "permfit_python"
  )
  results <- deep_py$permfit(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        conditional = FALSE,
        k_fold = 2L,
        index_i = index_i,
        n_perm = n_perm,
        n_jobs = n_jobs,
        backend = backend,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack
  )

  return(data.frame(
        method = "Permfit-DNN",
        importance = results$importance,
        p_value = results$pval,
        score = results$score
  ))
}


compute_dnn_py_cond <- function(sim_data,
                                index_i,
                                n = 1000L,
                                prob_type = "regression",
                                depth = 2,
                                n_perm = 100,
                                n_jobs = 1,
                                backend = NULL,
                                perm = FALSE,
                                nominal = NULL,
                                list_grps = list(),
                                group_stack = FALSE,
                                ...) {
  print("Applying DNN Conditional Method")
  
  deep_py <- import_from_path("permfit_py",
        path = "permfit_python"
  )
  results <- deep_py$permfit(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        max_depth = depth,
        n_perm = n_perm,
        n_jobs = n_jobs,
        backend = backend,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack,
        noPerm = no_perm
  )

  return(data.frame(
        method = "CPI-DNN",
        importance = results$importance,
        p_value = results$pval,
        score = results$score
  ))
}


compute_rf_cond <- function(sim_data,
                            index_i,
                            n = 1000L,
                            prob_type = "regression",
                            depth = 2,
                            n_perm = 100,
                            n_jobs = 1,
                            backend = NULL,
                            perm = FALSE,
                            nominal = NULL,
                            list_grps = list(),
                            group_stack = FALSE,
                            ...) {
  print("Applying DNN Conditional Method")
  
  deep_py <- import_from_path("permfit_py_RF",
        path = "permfit_python"
  )
  results <- deep_py$permfit_RF(
        X_train = sim_data[, -1],
        y_train = as.numeric(sim_data$y),
        prob_type = prob_type,
        index_i = index_i,
        conditional = TRUE,
        k_fold = 2L,
        max_depth = depth,
        n_perm = n_perm,
        n_jobs = n_jobs,
        backend = backend,
        list_nominal = nominal,
        groups = list_grps,
        group_stacking = group_stack
  )

  return(data.frame(
        method = "CPI-RF",
        importance = results$importance,
        p_value = results$pval,
        score = results$score
  ))
}


compute_grps <- function(sim_data,
                         prob_type = "regression",
                         num_trees = 2000L, 
                         list_grps = list(),
                         func = "gpfi",
                         ...) {
  task <- makeRegrTask(data = sim_data, target = "y")
  # RF model
  learner <- makeLearner("regr.ranger", par.vals = list(num.trees = num_trees))
  mod <- train(learner = learner, task = task)
  if (prob_type == "regression")
    res <- resample(learner, task, cv5, measures = mse, models = TRUE)
  gimp <- Gimp$new(task = task, res = res, mod = mod, lrn = learner)
  
  group <- c()
  for (i in 1:length(list_grps))
    group <- c(group, rep(paste0("G", i), length(list_grps[[i]])))
  group_df <- data.frame(feature = colnames(sim_data[, -1]), group = group, stringsAsFactors = FALSE)
  if (prob_type == "regression") {
    if (func == "gpfi") {
      res <- gimp$group_permutation_feat_imp(group_df, PIMP = FALSE, n.feat.perm = 100, regr.measure = mse)
      list_grps_new <- list()
      for (grp_ind in 1:length(list_grps)) {
        curr_grp <- ""
        for (i in list_grps[[grp_ind]])
          curr_grp <- paste(c(curr_grp, i), collapse=",")
        list_grps_new[[substring(curr_grp, 2)]] <- grp_ind
      }
      
      for (i in 1:dim(res)[1]) {
        res$features[i] <- list_grps_new[[res$features[i]]]
      }
      res <- res[mixedorder(as.character(res$features)), ]
      imp <- res$mse
    }
    if (func == "gopfi") {
      res <- gimp$group_only_permutation_feat_imp(group_df, PIMP = FALSE, n.feat.perm = 100, regr.measure = mse)
      res <- res[mixedorder(as.character(res$group_id)), ]
      res <- res[-1, ]
      imp <- res$GOPFI
    }
    if (func == "dgi") {
      res_list <- gimp$drop_group_importance(group_df, measures = mse)
      res <- c()
      for (i in 1:length(list_grps)) {
        res <- c(res, res_list[[i]]$aggr - res_list$all$aggr)
      }
      imp <- res
    }
    if (func == "goi") {
      res_list <- gimp$group_only_importance(group_df, measures = mse)
      res <- c()
      for (i in 1:length(list_grps)) {
        res <- c(res, res_list$featureless$aggr - res_list[[i]]$aggr)
      }
      imp <- res
    }
  }
  return(data.frame(
    method = func,
    importance = imp,
    p_value = NA,
    score = NA
  ))
}


# compute_vimp <- function(sim_data,
#                          seed,
#                          prob_type = "regression",
#                          ...) {
#     learners.lib <- c("SL.randomForest")
#     # learners.lib <- c("SL.mean", "SL.glmnet")

#     indx <- 1
#     y <- as.numeric(sim_data$y)
#     x <- sim_data[, -1]
#     vimp_imp <- c()
#     vimp_pval <- c()
#     for (indx in 1:dim(x)[2]) {
#         print(indx)
#         if (prob_type == "regression")
#             vimp <- vimp_rsquared(Y = y, X = x, indx = indx, SL.library = learners.lib,
#                           na.rm = TRUE, V = 2, cvControl = list(V = 2))
#         else
#             vimp <- vimp_auc(Y = y, X = x, indx = indx, SL.library = learners.lib,
#                              na.rm = TRUE, V = 2, cvControl = list(V = 2))
#         vimp_imp <- c(vimp_imp, vimp$est)
#         vimp_pval <- c(vimp_pval, vimp$p_value)
#     }
#     return(data.frame(
#         method = 'vimp',
#         importance = vimp_imp,
#         p_value = vimp_pval,
#         score = NA
#         ))
# }

compute_vimp <- function(sim_data,
                         seed,
                         prob_type = "regression",
                         ...) {
  print("Applying Vimp Method")
  utilspy <- import_from_path("utils_py",
                              path = "utils"
  )
  imp <- utilspy$compute_vimp_dnn(
    as.matrix(sim_data[, -1]),
    np$array(as.numeric(sim_data$y)),
    prob_type=prob_type,
    seed=seed)
  
  return(data.frame(
    method = 'vimp',
    importance = unlist(imp$val_imp),
    p_value = unlist(imp$p_value),
    score = NA
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

compute_cpi <- function(sim_data,
                        prob_type = "regression",
                        ...) {
  print("Applying CPI/Knockoff Method")
  library(mlr3)
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
    prob_type
  )
  # for(i in imp){
  #     print(i)
  # }
  # stop()
  # print(t.test(unlist(imp), alternative='greater')$p.value)
  # stop()
  return(data.frame(
    method = 'loco',
    importance = unlist(imp$val_imp),
    p_value = unlist(imp$p_value),
    score = NA
  ))
}