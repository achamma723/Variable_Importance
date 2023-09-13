suppressMessages({
  library("data.table")
  library("mvtnorm")
  # library("sigmoid")
  library
})


generate_cov_simple <- function(p,
                                rho = 0.5,
                                type = "toeplitz",
                                ...) {
  if (type == "toeplitz") {
    powers <- 0:(p - 1)
    sigma <- toeplitz(rho^powers)
  }
  if (type == "fixed") {
    sigma <- diag(p) * (1 - rho) +
      matrix(rho, p, p)
  }
  return(sigma)
}


generate_cov_blocks <- function(p,
                                rho = 0.5,
                                rho_group = 0.5,
                                n_blocks = 10L,
                                type = "toeplitz",
                                ...) {
  # Initialization of the Covariance matrix
  sigma <- matrix(0, p, p)
  # Compute the number of features per block
  p_per_block <- round(p / n_blocks)
  # If the rho parameter is not a list for the blocks
  # (inter-correlation)
  if (length(rho) != n_blocks)
    rho <- rep(rho, n_blocks)

  blocks_ind <- list()
  # Each block is associated with the simple covariance matrix
  for (block in 1:n_blocks) {
    if (block == n_blocks) {
      indx_interv <- ((block - 1) * p_per_block + 1):p
    } else {
      indx_interv <- ((block - 1) * p_per_block + 1):(block * p_per_block)
    }
    blocks_ind[[block]] <- indx_interv
  }

  for (block in 1:length(blocks_ind)) {
    sigma[blocks_ind[[block]], blocks_ind[[block]]] <-
      generate_cov_simple(length(indx_interv), rho[block], type = type)
    if (type == "group") {
      for (block_inter in 1:dim(rho_group)[2]) {
        if (block_inter != block)
          sigma[blocks_ind[[block]], blocks_ind[[block_inter]]] <- rho_group[block, block_inter]
      }
    }
  }
  return(sigma)
}


generate_data <- function(seed = 2021L,
                          file = "", # File of real data
                          sigma = "",
                          x = NULL,
                          n = 1000L,
                          p = 50L,
                          n_signal = 20L,
                          rho = 0.5,
                          rho_group = 0.5,
                          mean = 0.0,
                          n_blocks = 10L,
                          type_sim = "simple_toeplitz",
                          # It can be ["blocks_toeplitz",
                          # "blocks_fixed", "simple_toeplitz",
                          # "simple_fixed", "group_inter"]
                          snr = 3.6,
                          prob_sim_data = "regression",
                          effectset = c(
                            -0.5, -1, -2, -3,
                            0.5, 1, 2, 3
                          ),
                          is_file = FALSE,
                          ...) {
  if (!(file == "")) {
    x <- as.matrix(data.table::fread(file = paste0(file, ".csv")))
    res <- generate_data(
      x = x,
      seed = seed,
      n_signal = n_signal,
      prob_sim_data = prob_sim_data,
      is_file = TRUE,
    )
    return(res)
  }

  # Fix the seed for the random generator
  set.seed(seed)
  # The independent or correlated scenario
  independence <- ifelse(rho > 0, FALSE, TRUE)

  # Generating the design matrix Check if the input data is given
  if (is.null(x)) {
    # Check if sigma is given as input (covariance of a real dataset)
    if (sigma == "") {
      sigma <- diag(p)

      if (independence == FALSE) {
        # List of functions to generate the covariance matrix
        func_sim <- c(generate_cov_blocks, generate_cov_simple)
        names(func_sim) <- c("blocks", "simple")
        # Define the function to use according to the type in the
        # input
        type_split <- strsplit(as.character(type_sim), "_")[[1]]
        sigma <- func_sim[[type_split[1]]](p = p,
          rho = rho,
          rho_group = rho_group,
          n_blocks = n_blocks,
          type = type_split[2])
      }

      if (length(mean) != dim(sigma)[1]) {
      mean <- rep(mean, dim(sigma)[1])
      }

      x <- mvtnorm::rmvnorm(
      n = n,
      mean = mean,
      sigma = sigma,
      method = "chol"
      )
    }
    else {
      x <- reg_py$resample_correlated(sigma, seed = seed)
    }
    if (independence && (sigma == "")) x <- apply(x, 2, sample)
  }

  # Randomly draw n_signal predictors which are defined as signal
  # predictors
  predno <- sample(dim(x)[2], size = n_signal)

  # Reorder data matrix so that first n_signal predictors are the
  # signal predictors
  x_1 <- data.frame(x[, c(predno, which(!1:ncol(x) %in% predno))])

  # Determine beta coefficients
  beta <- effectset[sample(length(effectset),
    size = n_signal,
    replace = TRUE
  )]

  beta1 <- effectset[sample(length(effectset),
    size = choose(n_signal, 2),
    replace = TRUE
  )]

  # Generate response
  ## The product of the signal predictors with the beta coefficients
  prod_signal <- model.matrix(~ . + 0, x_1[, 1:n_signal]) %*% beta

  if (prob_sim_data == "classification") {
    y <- as.character(rbinom(
      n = dim(x_1)[1],
      size = 1,
      p = plogis(prod_signal)
    ))

    # Check if the classes are balanced
    while (min(table(y)) < 0.1 * dim(x_1)[1]) {
      y <- as.character(rbinom(
        n = dim(x_1)[1],
        size = 1,
        p = plogis(prod_signal)
      ))
    }
  }
  else {
    if (prob_sim_data == "regression_product") {
      prod_signal <- model.matrix(~ (.)^2 + 0 - ., x_1[, 1:n_signal]) %*% beta1
    }

    if (prob_sim_data == "regression_combine") {
      prod_signal <- model.matrix(~ (.)^2 + 0, x_1[, 1:n_signal]) %*% c(beta, beta1)
    }

    if (!(prob_sim_data %in% c("regression_perm", "regression_group_sim_1"))) {
      if (is_file == FALSE) {
        ## Computing the noise magnitude controlled by Signal-to-Noise ratio
        sigma_noise <- norm(prod_signal, type = "2") / (snr * sqrt(dim(x_1)[1]))
        y <- prod_signal + sigma_noise * rnorm(dim(x_1)[1])
      }
      else
        y <- prod_signal

      if (prob_sim_data == "regression_relu")
        y <- relu(y)
    }

    if (prob_sim_data == "regression_perm") {
      x_1 <- data.frame(x)
      y <- x_1[, 1] + 2 * log(1 + 2 * x_1[, 21]^2 + (x_1[, 41] + 1)^2) +
        x_1[, 61] * x_1[, 81] + rnorm(dim(x_1)[1])
    }
  }

  colnames(x_1) <- paste0("x", 1:dim(x_1)[2])
  return(data.frame(y, x_1))
}
