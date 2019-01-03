library(gbm)

traingbm = trainset
traingbm$y = ytrain

model = gbm(y~.,data = traingbm,
            distribution = 'multinomial',
            n.trees = 100,
            interaction.depth = 5,
            shrinkage = 0.01)
pred.prob <- predict(model, newdata = test, 
               n.trees = 100, 
               type="response")

pred <- apply(pred.prob, 1, which.max)

mean(pred == ytest)
###0.72333











