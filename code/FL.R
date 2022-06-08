# In this project, we call our simple Keras-like Image classifier FullyLight
# It simplifies procedure of Keras deep learning framework in Python, and can train
# the model, test model, and evaluate the model with FULLY NEURAL NETWORK,
# We implement its forward pass/backward pass manually
# All operations should be operated with double precision!
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/03
library(pryr)
library(assertthat)

#############################
#       Hyper-parameters    #
#############################
#dims <- c(256, 128, 3)
#hidden_activation <- 'relu'
#kernel_initializer <- 'xavier'#'zeros'#'ones' #never use 'ones' and 'zeros'!
#optimizer <- 'adam'
#metric <- 'acc'
learning_rate <- 1e-4
#epochs <- 10
batch_size <- 32
validation_rate <- 0.1
l1 <- 0.
l2 <- 0.
#############################
# Step 1 Load the data set
# as a test, use iris
# dataset <- iris
# # View(dataset)
# dim(dataset)
# # Step 2 Reshape the inputs
# # get x and y
# x <- dataset[, -dim(dataset)[2]]
# x <- as.matrix(x)
# num_classes <- length(unique(dataset[, dim(dataset)[2]]))
# num_classes
# y <- dataset[, dim(dataset)[2]]
# y <- as.factor(y)
# y <- as.vector(unclass(y))
# y <- to_categorical(y, num_classes = num_classes)
# y
FL <- function(data_name, dims, hidden_activation, 
               kernel_initializer, optimizer, metric, epochs,
               l1, l2, output = F){
  # This function is used to do fullylight test
  # Args :
  #   --data_name: data set name
  #   --dims: dimensions of NN
  #   --hidden_activation
  #   --kernel_initializer
  #   --optimizer/metric, epochs, l1, l2
  #   --output: default is F
  msg <- paste(c(data_name, '\n', dims, '\n', hidden_activation, '\n',
                 kernel_initializer, '\n', optimizer, '\n', 
                 metric, '\n', epochs, '\n', l1, '\n', l2, '\n'), collapse = '')
  message(msg)
  print(data_name)
  print(dims)
  print(hidden_activation)
  print(kernel_initializer)
  print(optimizer)
  print(metric)
  print(epochs)
  print(l1)
  print(l2)
  isimage <- F
  if (grepl('image', data_name, fixed = T)){ # image
    mode <- 'image'
    isimage <- T
    data_name <- tolower(str_split(data_name, '\\(')[[1]][1])
    data_path <- paste('../data/', data_name)
  } else if (grepl('structured', data_name, fixed = T)){ # structured
    mode <- 'structured'
    data_name <- tolower(str_split(data_name, '\\(')[[1]][1])
    data_path <- paste(c('../../data/', data_name, '.csv'), collapse = '')
  } else{ # random
    mode <- 'random'
  }
  # tell datasets
  if (metric == 'accuracy') 
    metric <- 'acc'
  if (hidden_activation == 'leakyrelu') hidden_activation <- 'prelu'
  sep <- ','
  if (data_name == 'iris' | data_name == 'scat') y_name <- 'Species'
  else if (data_name == 'mtcars') y_name <- 'am'
  useless_columns <- NULL
  if (data_name == 'churn') {
    y_name <- 'Churn'
    useless_columns <- c('customerID')
  }
  if (data_name == 'scat') useless_columns <- c('Month', 'Year')
  if (data_name == 'yeast') y_name <- 'X.class_protein_localization.'
  
  data_list <- data_preprocessing(data_path = data_path, mode = mode, useless_columns = useless_columns,
                                  sep = sep, y_name = y_name, data_name = data_name)
  x <- data_list$train_data
  y <- data_list$train_label
  # Step 3 Normalize the inputs
  # Step 5 Forward propagation(Vectorization/Activation functions)
  # Now, first define our model
  models <- FullyLight::Model(dims = dims, input_shape = ncol(x), hidden_activation = hidden_activation,
                  out_activation = 'softmax', kernel_initializer = kernel_initializer, l1 = l1, l2 = l2)
  # In particular, we implement this procedure using vectorization
  # then compile our model
  message(Summary(models))
  compile_funcs <- FullyLight::Compile(loss = 'categorical_crossentropy',
                           optimizer = optimizer, metric = metric, learning_rate = learning_rate, l1 = l1, l2 = l2)
  # Step 6 Compute cost
  # Step 7 Backward pass
  # Step 8 Update parameters
  # Now we start training
  res <- FullyLight::Fit(x = x, y = y, models = models, compile_funcs = compile_funcs,
             validation_rate = validation_rate, validation_data = NULL, epochs = epochs,
             batch_size = batch_size, verbose = 1, shuffle = T, isimage = isimage, data_name = data_name, output = output)
  # Step 9 Make a prediction
  eval_res <- FullyLight::Evaluate(x = data_list$test_data, y = data_list$test_label, models = res$models, params = res$params,
                       loss = 'categorical_crossentropy',
                       metric = metric)
  eval_res$eval_res
  eval_res$loss
}

# preds_res <- Predict(x = x, models = res$models, params = res$params, type = 'probs')
# preds_res
# plot(res$history$acc)
# 
# data <- data.frame(x1 = x[,1], x2 = x[, 2], y = apply(y, 1, function(x) which.max(x)))
# View(data)
# 
# View(preds_res)
# ggplot(data = data.frame(preds_res), aes(x = X1, y=X2)) + geom_point(color=apply(y, 1, function(x) which.max(x)))
