xgboost_cv <- function(train, lab, nrou, list_max.depth, list_eta, name, fold =5){
  
  
  num <- length(list_max.depth) * length(list_eta)
  vec <- rep("NA", num)
  mat <- matrix(NA, nrow = num, ncol = nrou)
  
  for(j in 1:length(list_max.depth)){
    for(k in 1:length(list_eta)){
      param <- list("objective" = "multi:softmax",
                    "num_class" = 4,
                    "eta" = list_eta[k], "max.depth" = list_max.depth[j])
      
      cv <- xgb.cv(data = train, label = lab, params = param , nround = nrou, nfold = fold, metrics = "merror", verbose = 0)
      pos <- (j-1)*length(list_eta)+k
      vec[pos] <- paste0("max.depth = ", list_max.depth[j], ", eta = ", list_eta[k])
      mat[pos, ] <- data.matrix(cv$evaluation_log)[, 4]
      
      
      # plot1 <- ggplot(cv$evaluation_log)+
      #   geom_line(aes(x = iter, y = test_merror_mean))+
      #   labs(title = paste0("max.depth = ", list_max.depth[j], ", eta = ", list_eta[k]))
      # print(plot1)
    }
  }
  
  error <- data.frame(cbind(vec, mat))
  colnames(error) <- c("cases", 1:nrou)
  
  
  library(reshape2)
  error_melt <- melt(error, id.vars = 'cases')
  plot1 <- ggplot(error_melt, mapping = aes(group = cases))+
            geom_smooth(aes(x = variable, y = round(as.numeric(value), 4), color = cases))+
            scale_x_discrete(breaks = seq(0, nrou, by = 10))+
            labs(title = paste0("XGBOOST & ", name), x = "nround", y = "Error Rate")
  
  return(list(error, plot1))
}
