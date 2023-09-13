## Object of class 'VarImp'


## as.VarImp
as.VarImp <- function(object, ...)
   UseMethod("as.VarImp")

as.VarImp.data.frame <- function(object, FUN = mean,
                             type = c("Permutation", "Conditional Permutation", "Selection Frequency", "See Info"), 
                             info = NULL, ...)
{
   match.fun(FUN)
   perror_mean <- apply(object, 1, mean)
   perror_std <- apply(object, 1, sd)
   z_test = perror_mean / perror_std
   p_val = 1 - stats::pnorm(z_test)
   out <- list(values = apply(object, 1, FUN, ...),
               p_val = p_val,
               perTree = object,
               type = match.arg(type),
               info = info)
   class(out) <- "VarImp"
   return(out)
}

as.VarImp.matrix <- function(object, FUN = mean,
                                 type = c("Permutation", "Conditional Permutation", "Selection Frequency", "See Info"), 
                                 info = NULL, ...)
{
   object <- as.data.frame(object)
   as.VarImp(object, FUN, type, info, ...)
}

as.VarImp.numeric <- function(object, perTree = NULL,
                              type = c("Permutation", "Conditional Permutation", "Selection Frequency", "See Info"), 
                              info = NULL, ...)
{
   out <- list(values = object,
               perTree = perTree,
               type = match.arg(type),
               info = info)
   class(out) <- "VarImp"
   return(out)
}

## is.VarImp
is.VarImp <- function(VarImp)
{
   names <- all(c("values", "perTree", "type", "info") %in% names(VarImp))
   all(names, class(VarImp) == "VarImp")
}

