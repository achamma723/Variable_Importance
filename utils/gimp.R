library(R6)
library(featureImportance)
library(checkmate)
library(testthat)
library(dplyr)
library(stringi)
library(future.apply)
library(mlr)

Gimp = R6Class("grouped_imp",
  public = list(
    initialize = function(task, res, mod, lrn) {
      private$.task = task
      private$.res = res
      private$.mod = mod
      private$.lrn = lrn
    },
    shapley = function(group_df, res, n.shapley.perm = 120L) {
      groups = unique(group_df$group)
      perm = featureImportance:::generatePermutations(groups, n.shapley.perm = n.shapley.perm)
      
      # generate all marginal contribution sets for each feature
      mc = lapply(groups, function(x) featureImportance:::generateMarginalContribution(x, perm))
      mc = unlist(mc, recursive = FALSE)
      # get all unique sets
      values = unique(unname(unlist(mc, recursive = FALSE)))
      values_feat = rep(list(NA), length(values))
      for (i in 1:length(values)) {
        values_feat[[i]] = group_df$feature[group_df$group %in% values[[i]]]
        if (length(values_feat[[i]]) == 0) values_feat[[i]] = values[[i]]
      }
      
      
      resampling = self$res
      res_list = list()
      for (i in 1:length(resampling$models)) {
        # compute value function based on importance
        value.function = lapply(values_feat, function(x) {
          calculateValueFunctionImportance(object = resampling$models[[i]], data = getTaskData(self$task)[resampling$pred$instance$test.inds[[i]], ], measures = getDefaultMeasure(self$task),
                                           n.feat.perm = 20, features = x)
        })
        
        vf = rbindlist(value.function)
        vf$features = stri_paste_list(values, ",")
        
        result_df = data.frame(group = groups, shapImp = NA)
        for(group in groups){
          mc = featureImportance:::generateMarginalContribution(group, perm)
          mc.val = featureImportance:::getMarginalContributionValues(mc, vf)
          imp = featureImportance:::getShapleyImportance(mc.val)
          result_df[result_df$group == group, ]$shapImp = as.numeric(imp)
        }
        res_list[[i]] = result_df
      }
      y = bind_cols(res_list)
      cols = !grepl(pattern = "group", x = colnames(y))
      data.frame(group = groups, meanShapImp = rowMeans(y[, cols]))
    },
    group_permutation_feat_imp = function(group_df, PIMP = TRUE, s = 10, n.feat.perm = 10L, regr.measure = mse) {
      checkmate::assert_data_frame(group_df)
      testthat::expect_equal(colnames(group_df), c("feature", "group"))
      checkmate::assert_subset(as.character(group_df$feature), self$features)
      gfeats = list()
      groups = unique(group_df$group)
      groups = groups[!is.na(groups)]
      for (i in 1:length(groups)) {
        gfeats[[i]] = group_df %>% dplyr::filter(group == unique(group_df$group)[i]) %>% pull(feature) %>% as.character()
      }
      names(gfeats) = groups
      imp = featureImportance::featureImportance(self$res, data = getTaskData(self$task), n.feat.perm = n.feat.perm,
        features = gfeats, measures = switch(getTaskType(self$task), classif = mmce, regr = regr.measure), local = FALSE,
        importance.fun = function(permuted, unpermuted) permuted - unpermuted)
      
      if (PIMP) {
        res = self$res
        task = self$task
        res = future_lapply(1:s, function(i) {
          PIMP_data = getTaskData(task)
          PIMP_data[getTaskTargetNames(task)] = sample(PIMP_data[getTaskTargetNames(task)][[1]])
          PIMP_imp = featureImportance::featureImportance(res, data = PIMP_data, n.feat.perm = n.feat.perm,
            features = gfeats, measures = switch(getTaskType(task), classif = mmce, regr = regr.measure), local = FALSE,
            importance.fun = function(permuted, unpermuted) permuted - unpermuted)
          data.frame(summary(PIMP_imp), i = i)
        })
        res2 = bind_rows(res)
        
        PIMP_df = data.frame()
        groups = unique(group_df$group)
        for (group in groups) {
          curr_grp = ""
          for (i in gfeats[[group]]) {
            curr_grp = paste(c(curr_grp, i), collapse=",")
          }
          curr_grp = substring(curr_grp, 2)
          x = res2 %>% dplyr::filter(features == curr_grp) %>% pull(2)
          y = summary(imp) %>% dplyr::filter(features == curr_grp) %>% pull(2)
          library(MASS)
          groupFIT = fitdistr(x, "normal")
          mu = groupFIT$estimate[["mean"]]
          sd = groupFIT$estimate[["sd"]]
          #ks.test(x, y = "pnorm", mean = mu, sd = sd) #test for normality
          p_val = pnorm(y, mean = mu, sd = sd, lower.tail = FALSE)
          PIMP_df = bind_rows(PIMP_df, data.frame(features = group, p_val = p_val))
          if (group == "app") saveRDS(object = list(x = x, y = y), file = "PIMP_example.RDS")
        }
        
        return(left_join(summary(imp), PIMP_df))
      } else {
        return(summary(imp))
      }
    },
    group_only_permutation_feat_imp = function(group_df, regr.measure = mse, PIMP = TRUE, s = 10, n.feat.perm = 10L) {
      checkmate::assert_data_frame(group_df)
      testthat::expect_equal(colnames(group_df), c("feature", "group"))
      checkmate::assert_subset(as.character(group_df$feature), self$features)
      gfeats = list()
      groups = unique(group_df$group)
      groups = groups[!is.na(groups)]
      for (i in 1:length(groups)) {
        gfeats[[i]] = group_df %>% dplyr::filter(group %in% setdiff(unique(group_df$group), unique(group_df$group)[i])) %>% pull(feature) %>% as.character()
      }
      names(gfeats) = groups
      gfeats$all = unique(group_df$feature)
      imp = featureImportance::featureImportance(self$res, data = getTaskData(self$task), n.feat.perm = n.feat.perm,
        features = gfeats, measures = switch(getTaskType(self$task), classif = mmce, regr = regr.measure), local = FALSE, 
        importance.fun = function(permuted, unpermuted) return(permuted))
      
      imp2 = summary(imp)

      imp2$GOPFI = NA
      imp2$group_id = NA
      for(j in 1:nrow(imp2)){
        for (i in 1:length(gfeats)) {
          feats = gfeats[[i]]
          if(identical(imp2$features[j],paste(feats,collapse = ","))) imp2$group_id[j] = names(gfeats)[i]
          #imp2$group_id[which(imp2$features == feats)] =  names(gfeats)[i]
        }
      }
      
      for (group in groups) {
        imp2$GOPFI[which(imp2$group_id == group)] = (imp2 %>% dplyr::filter(group_id == "all") %>% pull(2)) - (imp2 %>% dplyr::filter(group_id == group) %>% pull(2)) 
      }

      if (PIMP) {
        res = self$res
        task = self$task
        res = lapply(1:s, function(i) {
          PIMP_data = getTaskData(task)
          PIMP_data[getTaskTargetNames(task)] = sample(PIMP_data[getTaskTargetNames(task)][[1]])
          PIMP_imp = featureImportance::featureImportance(self$res, data = PIMP_data, n.feat.perm = n.feat.perm,
            features = gfeats, measures = switch(getTaskType(self$task), classif = mmce, regr = regr.measure), local = FALSE, 
            importance.fun = function(permuted, unpermuted) return(permuted))
          df = data.frame(summary(PIMP_imp), i = i)
          df$GOPFI = NA
          for (group in groups) {
            df$GOPFI[which(group == df$features)] =(df %>% dplyr::filter(features == "all") %>% pull(2)) - (df %>% dplyr::filter(features == group) %>% pull(2)) 
          }   
          df
        })
        res2 = bind_rows(res)
        
        PIMP_df = data.frame()
        groups = unique(group_df$group)
        for (group in groups) {
          x = res2 %>% dplyr::filter(features == group) %>% pull(2)
          y = summary(imp) %>% dplyr::filter(features == group) %>% pull(2)
          library(MASS)
          groupFIT = fitdistr(x, "normal")
          mu = groupFIT$estimate[["mean"]]
          sd = groupFIT$estimate[["sd"]]
          p_val = pnorm(y, mean = mu, sd = sd, lower.tail = TRUE) 
          PIMP_df = bind_rows(PIMP_df, data.frame(features = group, p_val = p_val))
        }
        
        return(left_join(imp2, PIMP_df))
      } else {
        return(imp2)
      }
    },
    drop_group_importance = function(group_df, resampling = cv10, measures = mse) {
      checkmate::assert_data_frame(group_df)
      testthat::expect_equal(colnames(group_df), c("feature", "group"))
      checkmate::assert_subset(as.character(group_df$feature), self$features)
      gfeats = list()
      groups = unique(group_df$group)
      groups = groups[!is.na(groups)]
      for (i in 1:length(groups)) {
        gfeats[[i]] = group_df %>% dplyr::filter(group %in% setdiff(unique(group_df$group), unique(group_df$group)[i])) %>% pull(feature) %>% as.character()
      }
      names(gfeats) = groups
      gfeats$all = unique(group_df$feature)
      
      res = list()
      data_all = getTaskData(self$task)
      res_inst = makeResampleInstance(resampling, task = self$task)
      for (group in c(groups, "all")) {
        data_group = data_all %>% dplyr::select(gfeats[[group]], getTaskTargetNames(self$task))
        task_group = makeRegrTask(data = data_group, target = getTaskTargetNames(self$task))
        res_group = resample(learner = self$lrn, task = task_group, measures = measures, resampling = res_inst)
        print(group)
        res[[group]] = res_group
      }
      res
    },
    group_only_importance = function(group_df, resampling = cv10, measures = mse) {
      checkmate::assert_data_frame(group_df)
      testthat::expect_equal(colnames(group_df), c("feature", "group"))
      checkmate::assert_subset(as.character(group_df$feature), self$features)
      gfeats = list()
      groups = unique(group_df$group)
      groups = groups[!is.na(groups)]
      
      for (i in 1:length(groups)) {
        gfeats[[i]] = group_df %>% dplyr::filter(group == unique(group_df$group)[i]) %>% pull(feature) %>% as.character()
      }
      names(gfeats) = groups
      
      res = list()
      data_all = getTaskData(self$task)
      res_inst = makeResampleInstance(resampling, task = self$task)
      for (group in groups) {
        data_group = data_all %>% dplyr::select(gfeats[[group]], getTaskTargetNames(self$task))
        task_group = makeRegrTask(data = data_group, target = getTaskTargetNames(self$task))
        res_group = resample(learner = self$lrn, task = task_group, measures = measures, resampling = res_inst)
        print(group)
        res[[group]] = res_group
      }
      res[["featureless"]] =  resample(learner = makeLearner("regr.featureless"), task = self$task, measures = measures, resampling = resampling)
      res
    },
    group_pdp = function(features, parts = 4) {
      df_features = data.frame(feature = names(features), variable = paste0("x", 1:length(features)))
      data = getTaskData(self$task)
      library(ggplot2)
      library(gridExtra)
      mod = self$mod
      task = self$task
      pdp = data.frame()
      for (i in 1:nrow(data)) { 
        data_pdp = data
        data_pdp[-i, as.character(names(features))] = data[i, as.character(names(features))]
        pred = predict(mod, newdata = data_pdp)
        resp = switch(getTaskType(task), classif = pred$data[[1]], regr = pred$data$response)
        pdp_i = data.frame(index = i, pd = mean(resp))
        pdp_i$dim_red = (data %>% scale() %>% data.frame() %>% slice(i) %>% dplyr::select(names(features)) %>% as.matrix()) %*% features
        pdp = bind_rows(pdp, pdp_i)
        print(round(i / nrow(data), 2))
      }
      
      dim_red = as.character(round(features, 2))
      dim_red = unlist(lapply(dim_red, function(i) ifelse(substr(i, 1, 1) == "-", i, paste0(" + ", i))))
      dim_red = gsub("-", " - ", dim_red)
      dim_red = paste0(dim_red, df_features$variable, collapse = "")
      if (substr(dim_red, 1, 3) == " + ") dim_red = substr(dim_red, 4, nchar(dim_red))
      
      p1 = ggplot(data = pdp, mapping = aes(y = pd, x = dim_red)) + geom_point()
      p1 = p1 + ylab("mean prediction") + xlab(dim_red)
      tb = paste0(df_features$variable, " = ",df_features$feature, collapse = "\n")
      # p1 = p1 + annotate(geom = "label", x = x, y = y, label = tb, hjust = 0, fill = "white")
      #spiderplot
      cuts = seq(min(pdp$dim_red), max(pdp$dim_red), length.out = parts + 1)
      pdp$part = cut(pdp$dim_red, breaks = cuts, include.lowest = TRUE)
      
      ggplots = list()
      for (i in 1:parts) {
        pdp_i = pdp %>% dplyr::filter(part == levels(unique(pdp$part))[i])
        if (nrow(pdp_i) == 0) {
          ggplots[[i]] = ggplot() + theme_void()
          next
        }
        data_i = data[pdp_i$index, names(features)]
        for (feat in names(data_i)) {
          spider_i = setNames(data.frame(c(
            max(data[[feat]], na.rm = TRUE), 
            min(data[[feat]], na.rm = TRUE), 
            mean(data_i[[feat]], na.rm = TRUE)
          )), feat)
          if (feat == names(data_i)[1]) {
            spider_df = spider_i
          } else {
            spider_df = data.frame(spider_df, spider_i)
          }
        }
        normalize = function(x) {(x - min(x)) / (max(x) - min(x))}
        spider_df2 = data.frame(t(apply(spider_df, 2, normalize)[3, ]))
        colnames(spider_df2) = df_features$variable
        library(ggradar)
        spider_df2 = data.frame(group = "group", spider_df2)
        
        gg = ggradar(spider_df2,
          # font.radar = "Helvetica",
          grid.label.size = 2,
          axis.label.size = 3, 
          group.point.size = 2,
          group.line.width = 1,
          plot.legend = FALSE
        ) 
        
        gg = gridExtra::grid.arrange(gg, 
          bottom = grid::textGrob(paste0("I = ", gsub(",", ", ", levels(unique(pdp$part))[i]), ", n = ", nrow(pdp_i)),
            gp = grid::gpar(fontsize = 6), vjust = -8))
        ggplots[[i]] = gg
      }
      
      gg_all = grid.arrange(arrangeGrob(grobs = ggplots, ncol = parts),
        p1 +  theme(plot.margin = unit(c(-15,0,0,0), "mm")),
        nrow = 2)
      return(list(gg_all = gg_all, ggplots = ggplots, p1 = p1, pdp = pdp))
    },
    group_pdp2 = function(features, parts = 10, x = 4, y = -0.08) {
      df_features = data.frame(feature = names(features), variable = paste0("x", 1:length(features)))
      data = getTaskData(self$task)
      library(ggplot2)
      library(gridExtra)
      mod = self$mod
      task = self$task
      pdp = data.frame()
      if (all(features < 0)) features = features * -1
      
      for (i in 1:nrow(data)) { 
        data_pdp = data
        data_pdp[-i, as.character(names(features))] = data[i, as.character(names(features))]
        pred = predict(mod, newdata = data_pdp)
        resp = switch(getTaskType(task), classif = pred$data[[1]], regr = pred$data$response)
        pdp_i = data.frame(index = i, pd = mean(resp))
        pdp_i$dim_red = (data %>% scale() %>% data.frame() %>% slice(i) %>% dplyr::select(names(features)) %>% as.matrix()) %*% features
        pdp = bind_rows(pdp, pdp_i)
        print(round(i / nrow(data), 2))
      }
      
      dim_red = as.character(round(features, 2))
      dim_red = unlist(lapply(dim_red, function(i) ifelse(substr(i, 1, 1) == "-", i, paste0(" + ", i))))
      dim_red = gsub("-", " - ", dim_red)
      dim_red = paste0(dim_red, df_features$variable, collapse = "")
      if (substr(dim_red, 1, 3) == " + ") dim_red = substr(dim_red, 4, nchar(dim_red))
      
      p1 = ggplot(data = pdp, mapping = aes(y = pd, x = dim_red)) + geom_point()
      p1 = p1 + ylab("mean prediction") + xlab(dim_red)
      tb = paste0(df_features$variable, " = ",df_features$feature, collapse = "\n")
      p1 = p1 + annotate(geom = "label", x = x, y = y, label = tb, hjust = 0, fill = "white")
      
      return(list(p1 = p1, pdp = pdp))
    }
  ),
  private = list(
    .task = NULL,
    .res = NULL,
    .feat_imp = NULL,
    .mod = NULL,
    .lrn = NULL
  ),
  active = list(
    mod = function() private$.mod,
    task = function() private$.task,
    lrn = function() private$.lrn,
    features = function() {
      return(getTaskFeatureNames(private$.task))
    },
    res = function() private$.res
  )
)
