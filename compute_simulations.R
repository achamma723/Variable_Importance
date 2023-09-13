DEBUG <- FALSE
N_SIMULATIONS <- `if`(!DEBUG, 1L:100L, 1L)
N_CPU <- ifelse(!DEBUG, 100L, 1L)

suppressMessages({
  require(data.table)
  if (!DEBUG) {
    require(snowfall)
    sfInit(parallel = TRUE, cpus = N_CPU, type = "SOCK")
    sfLibrary(cpi)
    sfLibrary(gtools)
    sfLibrary(mlr3learners)
    sfLibrary(party)
    sfLibrary(permimp)
    sfLibrary(reticulate)
    sfLibrary(snowfall)
    sfSource("data/data_gen.R")
    sfSource("utils/compute_methods.R")
  } else {
    library(cpi)
    library(mlr3learners)
    library(gtools)
    library("party", quietly = TRUE)
    library(permimp)
    library(reticulate)
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
  "marginal",
  "d0crt",
  "permfit",
  "cpi",
  "cpi_rf",
  "lazy",
  "cpi_knockoff",
  "loco",
  "strobl",
  # "loco_dnn"
  "knockoff",
  "shap",
  "sage",
  "mdi",
  "bart"
)
list_models <- paste0("Best_model_1_", N_SIMULATIONS)

##### Configuration #####

param_grid <- expand.grid(
  # File, if given, for the real data
  file = "",
  # The file to regenerate samples with same covariance, if given
  sigma = "",
  # The number of samples
  n_samples = ifelse(!DEBUG, 100L, 100L),
  # n_samples = `if`(!DEBUG, seq(100, 100, by = 100), 10),

  # The number of covariates
  n_features = ifelse(!DEBUG, 5L, 5L),
  # The number of relevant covariates
  n_signal = ifelse(!DEBUG, 2L, 2L),
  # The mean for the simulation
  mean = c(0),
  # The correlation coefficient
  rho = c(
    # 0,
    # 0.2,
    # 0.5,
    0.8
  ),
  # Number of blocks
  n_blocks = ifelse(!DEBUG, 1L, 1L),
  # Type of simulation
  # It can be ["blocks_toeplitz", "blocks_fixed",
  # "simple_toeplitz", "simple_fixed"]
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
  # The statistic to use with the knockoff method
  stat_knockoff = c(
    # "l1_regu_path",
    "lasso_cv",
    "bart",
    "deep"
  ),
  # Number of permutations/samples for the DNN algos
  n_perm = c(100L),
  # Number of cpus for the multiprocessing unit
  n_jobs = c(1L)
)

param_grid <- param_grid[
  ((!param_grid$scaled_statistics) & # if scaled stats
     (param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]])) & # and defaults
     (!param_grid$refit) & # and refit
     (!param_grid$method %in% c(
       "d0crt",
       "knockoff"
     ))) |
    ((!param_grid$scaled_statistics) & # or scaled
       (!param_grid$refit) &
       (param_grid$method == "knockoff")) |
    ((param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]])) &
       (param_grid$method == "d0crt")) |
    ((!param_grid$scaled_statistics) &
       (!param_grid$refit) &
       (param_grid$stat_knockoff %in% c(param_grid$stat_knockoff[[1]]))),
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
    sim_data <- generate_data(
      seed,
      ...
    )
    print("Done loading data!")
    
    timing <- system.time(
      out <- switch(as.character(method),
                    marginal = compute_marginal(
                      sim_data,
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
                    permfit = compute_permfit(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      ...
                    ),
                    cpi = compute_cpi(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      ...
                    ),
                    cpi_rf = compute_cpi_rf(
                      sim_data,
                      seed,
                      nominal = if (list(...)$sigma != "") read.csv(paste0(list(...)$sigma, "_nominal_columns.csv"))$x else "",
                      ...
                    ),
                    lazy = compute_lazy(
                      sim_data,
                      ...
                    ),
                    cpi_knockoff = compute_cpi_knockoff(
                      sim_data,
                      ...
                    ),
                    loco = compute_loco(
                      sim_data,
                      dnn = FALSE,
                      ntree = 100L,
                      ...
                    ),
                    loco_dnn = compute_loco(
                      sim_data,
                      dnn=TRUE,
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
          n_signal = n_signal,
          mean = mean,
          rho = rho,
          sigma = sigma,
          n_blocks = n_blocks,
          type_sim = type_sim,
          snr = snr,
          method = method,
          index_i = index_i,
          n_simulations = N_SIMULATIONS,
          stat_knockoff = stat_knockoff,
          refit = refit,
          scaled_statistics = scaled_statistics,
          prob_sim_data = prob_sim_data,
          prob_type = strsplit(as.character(prob_sim_data), "_")[[1]][1],
          n_perm = n_perm,
          n_jobs = n_jobs
        )
      )
    }
  )

results <- rbindlist(results, fill=TRUE)

out_fname <- paste0(getwd(), "/results/results_csv/", "simulation_results_blocks_100_allMethods.csv")

if (DEBUG) {
  out_fname <- gsub("\\.csv", "-debug.csv", out_fname)
}

fwrite(results, out_fname)

if (!DEBUG) {
  sfStop()
}