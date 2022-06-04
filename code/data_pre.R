# This script is used for data preprocessing
# We created a function for data preprocessing
# for 1. image with simple normalization
# 2. structured data with simple preprocessing
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/17
library(purrr)
library(keras)

data_preprocessing <- function(data_path, mode,
                               sep = NULL, y_name = NULL, data_name = NULL, useless_columns = NULL,
                               ...){
  # This function is used to implement data preprocessing, for image, we support
  # uploading train and test separately, for structured data, we support only whole
  # data set, we will split them in the function, for random, we create data randomly
  # Args :
  #   --data_path: data path
  #   --mode: 'image' or 'structured' or 'random'
  #   --sep: seperator for each cell
  #   --y_name: y's name
  stopifnot(mode %in% c('image', 'structured', 'random'))
  if (mode == 'structured'){ # for structured data
    pre_data <- read.csv2(data_path, header = T, sep = sep,
                          row.names = NULL)
    message(paste(names(pre_data), '\n', collapse = ''))
    if (!is.null(useless_columns))
      pre_data <- pre_data %>% dplyr::select(-useless_columns)

    # Then we start to do data preprocessing
    # 1. We first check if there are missing values in data set
    # in order to avoid data snooping problem, we do this on training set
    # and tackle this problem on test using statistics derived from training set
    # 2. then transform each column to computable form
    non_numeric_indices <- pre_data %>% sapply(function(x) !is.numeric(x)) %>% which() %>% names()
    # get each column unique value and map
    pre_data[, non_numeric_indices] <- pre_data %>% dplyr::select(non_numeric_indices) %>%
      map(., factor) %>% map(., unclass)
    # then we map NA values into one independent category for categorical columns
    # for numeric columns, we just replace NA with 0
    num_indices <- pre_data %>% sapply(is.numeric) %>% which() %>% names() # we get indices for each column
    pre_data[, num_indices] <- pre_data %>% dplyr::select(num_indices) %>%
      map(~replace(., is.na(.), 0))
    #if (y_name %in% non_numeric_indices) pre_data[[y_name]] <- pre_data[[y_name]] - 1

    # 3. as default for this project, if we need to do train/test split,
    # we do it with 9:1 split ratio
    train_indices <- createDataPartition(y = pre_data[[y_name]],
                                         p = 0.9, list = F)
    pre_data[[y_name]] <- unclass(as.factor(pre_data[[y_name]]))
    train_data <- pre_data[train_indices,]
    test_data <- pre_data[-train_indices,]

    # 4. Normalize the data set
    means <- colMeans(as.matrix(train_data %>% dplyr::select(-y_name)))
    stds <- as.matrix(train_data %>% dplyr::select(-y_name)) %>% apply(2, sd)
    num_classes <- length(unique(pre_data[[y_name]]))
    train_label <- train_data[[y_name]]
    train_label <- to_categoricalF(train_label, num_classes)
    test_label <- test_data[[y_name]]
    test_label <- to_categoricalF(test_label, num_classes)
    train_data <- train_data %>% dplyr::select(-y_name)
    test_data <- test_data %>% dplyr::select(-y_name)
    train_data <- sweep(train_data, 2, means, '-')
    train_data <- sweep(train_data, 2, stds, '/')
    test_data <- sweep(test_data, 2, means, '-')
    test_data <- sweep(test_data, 2, stds, '/')
    train_data <- as.matrix(sapply(train_data, as.numeric))
    # print(class(train_data))
    test_data <- as.matrix(sapply(test_data, as.numeric))

  } else if (mode == 'image'){ # for image
    # In this project, we support MNIST/FashionMNIST/CIFAR10
    stopifnot(data_name %in% c('mnist', 'fashionmnist', 'cifar10'))
    if (data_name %in% c('mnist', 'fashionmnist')){
      if (data_name == 'mnist')
        mnist <- dataset_mnist()
      else
        mnist <- dataset_fashion_mnist()
      train_data <- mnist$train$x#images
      indices <- sample(1 : dim(train_data)[1], size = 5000, replace = F)
      train_data <- train_data[indices,,]
      train_label <- mnist$train$y + 1#labels
      train_label <- train_label[indices]
      test_data <- mnist$test$x#images
      test_label <- mnist$test$y + 1#labels
      train_label <- to_categoricalF(train_label, 10)
      test_label <- to_categoricalF(test_label, 10)
      train_data <- train_data / 255.
      test_data <- test_data / 255.
      # final_train_data <- matrix(rep(0, 784 * dim(train_data)[1]), ncol = 784)
      # for (i in 1 : dim(train_data)[1]){
      #   temp <- c(train_data[i,,])
      #   final_train_data[i,] <- temp
      # }
      # final_test_data <- matrix(rep(0, 784 * dim(test_data)[1]), ncol = 784)
      # for (i in 1 : dim(test_data)[1]){
      #   temp <- c(test_data[i,,])
      #   final_test_data[i,] <- temp
      # }
      #train_data <- final_train_data
      #test_data <- final_test_data
      dim(train_data) <- c(dim(train_data)[1], 784)
      dim(test_data) <- c(dim(test_data)[1], 784) # flatten to vector
    } else{ # cifar10, use keras:: package
      # we omit
    }
  } else{ # random data
    # We generate 350 points for training, 50 for testing
    set.seed(2233)
    x1 <- runif(400, -1, 1)
    x2 <- runif(400, -1, 1)
    data <- data.frame(x1 = x1, x2 = x2)
    y_score <- x1 ** 2 + x2 ** 2
    y <- ifelse(y_score > 0.5, 1, 2)
    y <- to_categoricalF(y, 2)
    train_data <- as.matrix(data[1 : 350,])
    train_label <- as.matrix(y[1 : 350,])
    test_data <- as.matrix(data[351 : 400,])
    test_label <- as.matrix(y[351 : 400,])
  }
  return (list(train_data = train_data, test_data = test_data,
               train_label = train_label, test_label = test_label))
}

# unit test
# data_list <- data_preprocessing(data_path = '../data/churn.csv', mode = 'structured',
#                                sep = ',', y_name = 'Churn')
# dim(data_list$train_data)
# dim(data_list$train_label)
# dim(data_list$test_data)
# dim(data_list$test_label)


