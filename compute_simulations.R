DEBUG <- FALSE
N_SIMULATIONS <- `if`(!DEBUG, 1L:100L, 1L)
N_CPU <- ifelse(!DEBUG, 100L, 1L)

suppressMessages({
  require(data.table)
  if (!DEBUG) {
    require(snowfall)
    sfInit(parallel = TRUE, cpus = N_CPU, type = "SOCK")
    sfLibrary(snowfall)
    sfLibrary(doParallel)
    sfLibrary(grf)
    sfLibrary(party)
    sfLibrary(permimp)
    sfLibrary(ranger)
    sfLibrary(randomForest)
    sfLibrary(reticulate)
    sfLibrary(vita)
    sfLibrary(gtools)
    sfLibrary(deepTL)
    sfLibrary(cpi)
    sfLibrary(MASS)
    sfLibrary(mlr3)
    sfLibrary(mlr3learners)
    sfLibrary(simstudy)
    sfLibrary(SuperLearner)
    sfLibrary(vimp)
    sfLibrary(glmnet)
    sfLibrary(xgboost)
    sfSource("data/data_gen.R")
    sfSource("utils/compute_methods.R")
  } else {
    library(doParallel)
    library(grf)
    library("party", quietly = TRUE)
    library(permimp)
    library(ranger)
    library(randomForest)
    library(reticulate)
    library(vita)
    library(gtools)
    library(deepTL)
    library(cpi)
    library(MASS)
    library(mlr3)
    library(mlr3learners)
    library(simstudy)
    library(SuperLearner)
    library(vimp)
    library(glmnet)
    library(xgboost)
    source("data/data_gen.R")
    source("utils/compute_methods.R")
  }
})

my_apply <- lapply
if (!DEBUG) {
  my_apply <- sfLapply
}

##### Running Methods #####

methods <- c(
  # "marginal",
  # "knockoff",
  # "shap",
  # 'sage'
  # "mdi"
  # "d0crt"
  # "bart"
  # "dnn",
  # "dnn_py"
  # "dnn_py_cond"
  # "rf_cond",
  # "gpfi",
  # "gopfi",
  # "dgi",
  # "goi",
  # "vimp"
  # "lazy"
  # "cpi_knockoff",
  "loco"
  # "ale",
  # "strobl"
  # "janitza",
  # "altmann",
  # "bartpy",
  # "grf"
)
list_models <- paste0("Best_model_1_", N_SIMULATIONS)

##### Configuration #####

param_grid <- expand.grid(
  # File, if given, for the real data
  # file = "data/ukbb_data",
  file = "",
  # The file to regenerate samples with same covariance, if given
  # sigma = "data/ukbb_data_age_no_hot_encoding",
  sigma = "",
  # The number of samples
  n_samples = ifelse(!DEBUG, 1000L, 1000L), #8300 #1000
  # n_samples = `if`(!DEBUG, seq(100, 1000, by = 100), 10L),
  # The number of covariates
  n_features = ifelse(!DEBUG, 100L, 50L), #2300 #1500
  # Whether to use or not grouped variables
  group_bool = c(
    # TRUE
    FALSE
  ),
  # Whether to use the stacking method
  group_stack = c(
    # TRUE,
    FALSE
  ),
  # The number of relevant covariates
  n_signal = ifelse(!DEBUG, 20L, 2L), #115 #75
  # The mean for the simulation
  mean = c(0),
  # The correlation coefficient
  rho = c(
    # 0,
    # 0.2,
    # 0.5,
    0.8
  ),
  # The correlation between the groups if group-based simulations
  rho_group = c(
    0
    # 0.2,
    # 0.5,
    # 0.8
  ),
  # Number of blocks
  n_blocks = ifelse(!DEBUG, 10L, 10L),
  # Type of simulation
  # It can be ["blocks_toeplitz", "blocks_fixed",
  # "simple_toeplitz", "simple_fixed", "blocks_group"]
  type_sim = c("blocks_fixed"),
  # Signal-to-Noise ratio
  snr = c(4),
  # The task (computation of the response vector)
  prob_sim_data = c(
    # "classification"
    # "regression"
    "regression_combine"
    # "regression_product",
    # "regression_relu"
    # "regression_perm"
    # "regression_group_sim_1"
  ),
  # The running methods implemented
  method = methods,
  # The d0crt method'statistic tests scaled or not
  scaled_statistics = c(
    # TRUE,
    FALSE
  ),
  # Refit parameter for the d0crt method
  refit = FALSE,
  # The holdout importance's implementation (ranger or original)
  with_ranger = FALSE,
  # The holdout importance measure to use (impurity corrected vs MDA)
  with_impurity = FALSE,
  # The holdout importance in python
  with_python = FALSE,
  # The statistic to use with the knockoff method
  stat_knockoff = c(
    # "l1_regu_path",
    "lasso_cv"
    # "bart"
    # "deep"
  ),
  # Type of forest for grf package
  type_forest = c(
    "regression",
    "quantile"
  ),
  # Depth for the Random Forest (Conditional Sampling)
  # depth = c(1:10)
  depth = c(2L),
  # Number of permutations/samples for the DNN algos
  n_perm = c(100L),
  # Number of cpus for the multiprocessing unit
  n_jobs = c(1L),
  # Type of backend to use with the multiprocessing unit
  backend = c("multiprocessing"),
  # Permutation or random sampling for residuals (CPI-DNN)
  no_perm = c( TRUE
               # FALSE
  )
)

param_grid <- param_grid[
  ((!param_grid$scaled_statistics) & # if scaled stats
     (param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]])) & # and defaults
     (!param_grid$refit) & # and refit
     (param_grid$type_forest == "regression") & # and type_forest
     (!param_grid$method %in% c(
       "d0crt", # but not ...
       "knockoff",
       "grf"
     ))) |
    ((!param_grid$scaled_statistics) & # or scaled
       (!param_grid$refit) &
       (param_grid$type_forest == "regression") &
       (param_grid$method == "knockoff")) |
    ((param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]])) &
       (param_grid$type_forest == "regression") &
       (param_grid$method == "d0crt")) |
    ((!param_grid$scaled_statistics) &
       (!param_grid$refit) &
       (param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]])) &
       (param_grid$method == "grf")),
]

param_grid$index_i <- 1:nrow(param_grid)
cat(sprintf("Number of rows: %i \n", nrow(param_grid)))

if (!DEBUG) {
  # Models names for saving DNNs
  sfExport("list_models")
  sfExport("param_grid")
}

compute_method <- function(method,
                           index_i,
                           n_simulations, ...) {
  print("Begin")
  cat(sprintf("%s: %i \n", method, index_i))
  
  compute_fun <- function(seed, ...) {
    # sfCat(paste("Iteration: ", seed), sep="\n")
    sim_data <- generate_data(
      seed,
      ...
    )
    print("Done loading data!")
    
    # Prepare the list of grouped labels
    if (list(...)$group_bool) {
      list_grps <- generate_grps(list(...)$p, list(...)$n_blocks)
    }
    else
      list_grps <- list()
    
    timing <- system.time(
      out <- switch(as.character(method),
                    marginal = compute_marginal(
                      sim_data,
                      list_grps = list_grps,
                      ...
                    ),
                    ale = compute_ale(sim_data,
                                      ntree = 100L,
                                      ...
                    ),
                    knockoff = compute_knockoff(sim_data,
                                                seed,
                                                list_models[[seed]],
                                                verbose = TRUE,
                                                ...
                    ),
                    bart = compute_bart(sim_data,
                                        ntree = 100L,
                                        ...
                    ),
                    mdi = compute_mdi(sim_data,
                                      ntree = 500L,
                                      ...
                    ),
                    shap = compute_shap(sim_data,
                                        ntree = 100L,
                                        ...
                    ),
                    sage = compute_sage(sim_data,
                                        seed,
                                        ntree = 100L,
                                        ...
                    ),
                    strobl = compute_strobl(sim_data,
                                            ntree = 100L,
                                            conditional = TRUE,
                                            ...
                    ),
                    d0crt = compute_d0crt(sim_data,
                                          seed,
                                          loss = "least_square",
                                          statistic = "randomforest",
                                          ntree = 100L,
                                          verbose = TRUE,
                                          ...
                    ),
                    janitza = compute_janitza(sim_data,
                                              cv = 2L,
                                              ncores = 3L,
                                              ...
                    ),
                    altmann = compute_altmann(sim_data,
                                              nper = 100L,
                                              ...
                    ),
                    dnn = compute_dnn(
                      sim_data,
                      ...
                    ),
                    grf = compute_grf(
                      sim_data,
                      ...
                    ),
                    bartpy = compute_bart_py(
                      sim_data,
                      ...
                    ),
                    dnn_py = compute_dnn_py(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      list_grps = list_grps,
                      ...
                    ),
                    dnn_py_cond = compute_dnn_py_cond(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      list_grps = list_grps,
                      ...
                    ),
                    rf_cond = compute_rf_cond(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      list_grps = list_grps,
                      ...
                    ),
                    gpfi = compute_grps(
                      sim_data,
                      seed,
                      list_grps = list_grps,
                      func = "gpfi",
                      ...
                    ),
                    gopfi = compute_grps(
                      sim_data,
                      seed,
                      list_grps = list_grps,
                      func = "gopfi",
                      ...
                    ),
                    dgi = compute_grps(
                      sim_data,
                      seed,
                      list_grps = list_grps,
                      func = "dgi",
                      ...
                    ),
                    goi = compute_grps(
                      sim_data,
                      seed,
                      list_grps = list_grps,
                      func = "goi",
                      ...
                    ),
                    vimp = compute_vimp(
                      sim_data,
                      seed,
                      ...
                    ),
                    lazy = compute_lazy(
                      sim_data,
                      ...
                    ),
                    cpi_knockoff = compute_cpi(
                      sim_data,
                      ...
                    ),
                    loco = compute_loco(
                      sim_data,
                      ntree = 100L,
                      ...
                    )
      )
    )
    out <- data.frame(out)
    out$elapsed <- timing[[3]]
    out$correlation <- list(...)$rho
    out$correlation_group <- list(...)$rho_group
    out$n_samples <- list(...)$n
    out$prob_data <- list(...)$prob_sim_data
    out$group_based <- list(...)$group_bool
    out$group_stack <- list(...)$group_stack
    # sfCat(paste("Done Iteration: ", seed), sep="\n")
    return(out)
  }
  sim_range <- n_simulations
  # compute results
  result <- my_apply(sim_range, compute_fun, ...)
  # postprocess and package outputs
  result <- do.call(rbind, lapply(sim_range, function(ii) {
    out <- result[[ii - min(sim_range) + 1]]
    out$iteration <- ii
    out
  }))
  
  res <- data.table(result)[,
                            mean(elapsed),
                            by = .(
                              n_samples,
                              correlation,
                              method,
                              iteration,
                              prob_data
                            )
  ]
  
  res <- res[,
             sum(V1) / (N_CPU * 60),
             by = .(
               n_samples,
               method,
               correlation,
               prob_data
             )
  ]
  
  print(res)
  print("Finish")
  
  return(result)
}


# if (DEBUG) {
#   set.seed(42)
#   param_grid <- param_grid[sample(1:nrow(param_grid), 5), ]
# }

results <-
  by(
    param_grid, 1:nrow(param_grid),
    function(x) {
      with(
        x,
        compute_method(
          file = file,
          n = n_samples,
          p = n_features,
          group_bool = group_bool,
          group_stack = group_stack,
          n_signal = n_signal,
          mean = mean,
          rho = rho,
          rho_group = rho_group,
          sigma = sigma,
          n_blocks = n_blocks,
          type_sim = type_sim,
          snr = snr,
          method = method,
          index_i = index_i,
          n_simulations = N_SIMULATIONS,
          stat_knockoff = stat_knockoff,
          with_ranger = with_ranger,
          with_impurity = with_impurity,
          with_python = with_python,
          refit = refit,
          scaled_statistics = scaled_statistics,
          type_forest = type_forest,
          prob_sim_data = prob_sim_data,
          prob_type = strsplit(as.character(prob_sim_data), "_")[[1]][1],
          depth = depth,
          n_perm = n_perm,
          n_jobs = n_jobs,
          backend = backend,
          no_perm = no_perm
        )
      )
    }
  )

results <- rbindlist(results, fill=TRUE)
print(results)
stop()
out_fname <- "simulation_results_blocks_100_cpiknockoff_loco_cpiRF_100.csv"

if (DEBUG) {
  out_fname <- gsub(".csv", "-debug.csv", out_fname)
}

fwrite(results, out_fname)

if (!DEBUG) {
  sfStop()
}