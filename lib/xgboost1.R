setwd("~/GitHub/Spring2018-Project3-Group1/lib")

dat <- read.csv("../lib/gist/gist10.csv")
lab <- read.csv("../data/label_train.csv")



set.seed(1)
index <- sample(1:3000, 800, replace = FALSE)
train <- data.matrix(dat[index, ])
lab1 <- data.matrix(lab[index, 3])
lab1[which(lab1 == 3)] = 0

library(xgboost)

# dtrain <- xgb.DMatrix(data = train, label = lab1)
nrou <- 100
list_max.depth <- c(3, 5,10, 20)
list_eta <- c(0.03, 0.3, 0.8)
num <- length(list_max.depth) * length(list_eta)
vec <- rep("NA", num)
mat <- matrix(NA, nrow = num, ncol = nrou)

for(j in 1:length(list_max.depth)){
  for(k in 1:length(list_eta)){
param <- list("objective" = "multi:softmax",
              "num_class" = 3,
              "eta" = list_eta[k], "max.depth" = list_max.depth[j])

cv <- xgb.cv(data = train, label = lab1, params = param , nround = nrou, nfold = 5, metrics = "merror", verbose = 0)
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
ggplot(error_melt, mapping = aes(group = cases))+
  geom_line(aes(x = variable, y = value, color = cases))

print(cv)
print(cv, verbose = T)

library(ggplot2)
ggplot(cv$evaluation_log)+
  geom_line(aes(x = iter, y = test_merror_mean))

class(train)
class(lab1)
dim(train)
dim(lab1)

### 1 ####
control <- trainControl(method="cv", number = 5, search = "grid", verboseIter = TRUE, returnData = FALSE
                        , allowParallel = TRUE)

xgb_grid_1 = expand.grid(
  nrounds = 100,
  max_depth = c(5, 10),
  eta = c(0.1, 1, 10),
  gamma = c(0.1, 1, 10),
  colsample_bytree = c(0.1, 0.5, 1),
  min_child_weight = c(0.1, 1, 10),
  subsample = c(0.1, 0.5, 1)
)

model1 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_1, objecitve = "multi:softprob", tuneLength=3)

preds <- predict(model1, test[, -c(1,2)])
acc <- 1 - sum(preds != test[,2])/length(test[,2])

# Tune 1: CV Error 0.216, Test Error 0.232
# Tune 2: CV Error 0.2075599, Test Error 0.238667
# Tune 3: CV Error 0.2111046, Test Error 0.2306667
# Tune 4: CV Error 0.2186661, Test Error 0.22
# Tune 5: CV Error  0.2097821, Test Error = 0.2093333
# Tune 6: CV Error 0.2093459, Test err = 0.2186667
### might be starting to overfit here
# Tune 7: CV err = 0.2084235, Test err = 0.2146667

xgb_grid_7 = expand.grid(
  nrounds = 100,
  max_depth = c(16, 17, 18),
  eta = c(0.17, 0.18, 0.19),
  gamma = c(0.44, 0.45, 0.46),
  colsample_bytree = c(1),
  min_child_weight = c(6.4, 6.5, 6.6),
  subsample = c(0.6, 0.65, 0.7)
)

model7 <- train(x = train[, -c(1,2)], y = train[,2], method='xgbTree', trControl=control, tuneGrid = xgb_grid_7, objecitve = "multi:softprob", tuneLength=3)

1 - max(model7[["results"]][["Accuracy"]])
preds <- predict(model7, test[, -c(1,2)])
err <- sum(preds != test[,2])/length(test[,2])

# Best model is model 5
# max_depth = 18, eta = 0.18, gamma = 0.45, colsample_bytree = 1, min_child_weight = 6.6, subsample = 0.7 on full training set

xgbbest <- xgboost(data = data.matrix(train[,-c(1,2)]),
                   label = train[,1],
                   max_depth = 18,
                   eta = 0.18,
                   gamma = 0.45,
                   colsample_bytree = 1,
                   min_child_weight = 6.6,
                   subsample = 0.7,
                   nrounds = 100,
                   num_class = 3,
                   early_stopping_rounds = 100,
                   prediction = TRUE,
                   metrics = "merror",
                   objective = "multi:softmax")

preds <- predict(xgbbest, data.matrix(test[, -c(1,2)]))
err <- sum(preds != test[,1])/length(test[,1])

### 2 ###
xgb_train <- function(dat_train, label_train, max.depth = 5, nround = 100,run.cv=F){
  library(xgboost)
  
  train_df <- data.matrix(dat_train)
  label <- label_train
  #train_df$label <- label_train
  #train_df <- data.matrix(train_df)
  if(run.cv){
    list_max.depth <- c(5,20,50,100)
    #list_max.depth <- c(5,10)
    #list_eta <-  c()
    #list_nround <- c(10,20)
    list_nround <- c(10,25,100,200)
    
    errors <- matrix(NA,nrow=length(list_max.depth),ncol = length(list_nround))
    
    for(j in 1:length(list_max.depth)){
      for(k in 1:length(list_nround)){
        param <- list("objective" = "multi:softmax",
                      "num_class" = 3,
                      "eval_metric" = "merror",
                      "eta" = .01, "max.depth" = list_max.depth[j])
        errors[j,k] <- xgb.cv(train_df, label, params = param, nround = list_nround[k])
      }
    }
    
    row_index <- which(errors == min(errors), arr.ind = TRUE)[1]
    col_index <- which(errors == min(errors), arr.ind = TRUE)[2]
    
    best.max.depth <- list_max.depth[row_index]
    best.nround <- list_nround[col_index]
    
    print(errors)
    cat('best number of max depth is: ',best.max.depth)
    cat('\n')
    cat('best number of nround is: ', best.nround)
    
  } else{
    best.max.depth <- max.depth
    best.nround <- nround
    
  }
  
 

  best_xgb_fit <- xgboost(data = train_df, 
                          label = label_train, 
                          max.depth = best.max.depth, 
                          eta = 0.3, 
                          nround = best.nround,
                          objective = "multi:softmax",
                          num_class=3,
                          verbose = 0)
  
  return(best_xgb_fit)
  
}

test <- xgb_train(dat_train = train, label_train = lab1, run.cv = T)

