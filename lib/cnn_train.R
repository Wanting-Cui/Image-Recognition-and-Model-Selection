extract_feature <- function(img_dir, width =256, height = 256) {
  library(caTools)
  library(EBImage)
  library(pbapply)
  
  img_size = width*height
  images_names  = list.files(path=img_dir, pattern = ".*.jpg")
  
  print(paste("Start processing", length(images_names), "images"))
  feature_list <- pblapply(images_names, function(imgname) {
    
    img <- readImage(file.path(img_dir, imgname))
    img_resized <- resize(img, w = width, h = height)
    grayimg <- channel(img_resized, "gray")
    img_matrix = grayimg@.Data
    img_vector = as.vector(t(img_matrix))
    return(img_vector)
  })
  feature_matrix <- do.call(rbind, feature_list)  ## bind the list of vector into matrix
  feature_matrix <- as.data.frame(feature_matrix)
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  label = read.csv("/Users/warabe/Desktop/2018Spring/ads/train/label_train.csv", header = T)
  feature_matrix = cbind(label[,3],feature_matrix)
  
  # feature_matrix = cbind(sample.split(label[,2],SplitRatio=ratio),feature_matrix)
  # colnames(feature_matrix)[1:2] = c("is_train","label")
  
  # df_second = data.frame(matrix(NA, nrow = ratio*dim(feature_matrix)[1], ncol = (img_size+2)))
  # df_second[,1:2] = feature_matrix[feature_matrix$is_train==T,1:2]
  # 
  # for (i in 1:sum(feature_matrix$is_train==T)){
  #   df_second[i,3:4098] = unlist(t(flop(t(matrix(feature_matrix[feature_matrix$is_train==T,][i,3:4098],nrow = 64,ncol = 64)))))
  # }
  # names(df_second)[3:4098] = paste0("pixel", c(1:4096))
  # names(df_second)[1:2] = c("is_train","label")
  
  return(feature_matrix)
  
}

cnn_data = extract_feature('/Users/warabe/Desktop/2018Spring/ads/train/images/')

write.csv(cnn_data, "cnn_data.csv")


train_cnn = function (dat_train_cnn){
  require(mxnet)
  start_time_cnn = Sys.time()
  
  train_x = t(dat_train_cnn[-testlab,2:(128^2+1)])
  train_y = c(dat_train_cnn[-testlab,1])
  
  train_array = train_x
  dim(train_array) = c(128, 128, 1, ncol(train_x))
  
  
  test_x = t(dat_train_cnn[testlab,2:(128^2+1)])
  test_y = c(dat_train_cnn[testlab,1])
  
  test_array = test_x
  dim(test_array) = c(128, 128, 1, ncol(test_x))
  
  
  data <- mx.symbol.Variable('data')
  # 1st convolutional layer
  conv_1 = mx.symbol.Convolution(data = data, kernel = c(4, 4), num_filter = 32)
  act_1 = mx.symbol.Activation(data = conv_1, act_type = "relu")
  pool_1 = mx.symbol.Pooling(data = act_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  dropout_1 = mx.symbol.Dropout(data = pool_1, p=0.3)
  # 2nd convolutional layer
  conv_2 = mx.symbol.Convolution(data = dropout_1, kernel = c(4, 4), num_filter = 32)
  act_2 = mx.symbol.Activation(data = conv_2, act_type = "relu")
  pool_2 = mx.symbol.Pooling(data=act_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  dropout_2 = mx.symbol.Dropout(data = pool_2, p=0.3)
  
  # 3nd convolutional layer
  conv_3 = mx.symbol.Convolution(data = dropout_2, kernel = c(4, 4), num_filter = 64)
  act_3 = mx.symbol.Activation(data = conv_3, act_type = "relu")
  pool_3 = mx.symbol.Pooling(data=act_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  dropout_3 = mx.symbol.Dropout(data = pool_3, p=0.3)
  
  
  
  # 1st fully connected layer
  flatten = mx.symbol.Flatten(data = dropout_3)
  
  fc_1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
  nb_1 = mx.symbol.BatchNorm(data = fc_1)
  act_4 = mx.symbol.Activation(data = nb_1, act_type = "relu")
  dropout_4 = mx.symbol.Dropout(data = act_4, p=0.3)
  
  # 2nd fully connected layer
  fc_2 = mx.symbol.FullyConnected(data = dropout_4, num_hidden = 3)
  nb_2 = mx.symbol.BatchNorm(data = fc_2)
  
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.SoftmaxOutput(data = nb_2)
  
  mx.set.seed(100)
  devices <- mx.cpu()
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = train_array,
                                       y = train_y,
                                       ctx = devices,
                                       num.round = 60,
                                       array.batch.size = 64,
                                       learning.rate = 0.01,
                                       eval.metric = mx.metric.accuracy,
                                       initializer = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
                                       optimizer = "adam",
                                       #epoch.end.callback = mx.callback.log.train.metric(100),
                                       epoch.end.callback = mx.callback.save.checkpoint("train_cnn"))
  
  
  end_time_cnn = Sys.time() # Model End time
  cnn_time = end_time_cnn - start_time_cnn #Total Running Time
  return(model)
}

ddd <- train_cnn(cnn_data)

cnnpre = predict(ddd,)

testlab = read.table("/Users/warabe/Documents/GitHub/Spring2018-Project3-Group1/output/TEST-NUMBER.txt")
testlab = c(testlab$V1, testlab$V2, testlab$V3, testlab$V4, testlab$V5)


predcnn <- predict(model)

require(devtools)
install_version("DiagrammeR", version = "0.9.0", repos = "http://cran.us.r-project.org")
require(DiagrammeR)


test_x = t(cnn_data[testlab,2:4097])
test_y = c(cnn_data[testlab,1])

test_array = test_x
dim(test_array) = c(64, 64, 1, ncol(test_x))

predcnn <- predict(ddd, test_array, )





