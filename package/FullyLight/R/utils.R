# This script contains all utilities functions for project
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/04

library(caret)
library(glue)

##############################
#       one-hot encoding     #
##############################
#' Change y into categorical(one-hot encoding).
#'
#' Keras-like to_categorical
#' @param y vector. Contains ground-truth labels.
#' @param num_classes numeric. Number of classes.
#' @return One-hot encoding labels.
#' @examples
#' y <- c(1, 2, 3)
#' y_ <- to_categoricalF(y, 3)
#' @export
to_categoricalF <- function(y, num_classes){
  # one-hot encoding
  # Args :
  #   --y: original labels
  #   --num_classes
  # return :
  #   --y: output one-hot labels
  y_ <- matrix(rep(0, length(y) * num_classes), nrow = length(y))
  for (i in 1 : length(y)){
    yy <- y[i]
    y_[i, yy] <- 1 # one_hot
  }

  return (y_)
}
# unit test
# y <- c(1, 2, 5)
# to_categoricalF(y, 5)

##############################
#       Shuffle data         #
##############################
shuffle_data <- function(x, y){
  # Shuffling data set
  # Args :
  #   --x: input x, shape: [m, in_dim]
  #   --y: input labels, one-hot, shape: [m, out_dim]
  # return :
  #   --x and y
  random_indices <- sample(1 : nrow(x), size = nrow(x), replace = F)
  x <- x[random_indices, ]
  y <- y[random_indices, ]

  return (list(x = x, y = y))
}

# unit test
# x <- matrix(c(3, 4, 5, 6, 7, 8, 1, 2, 3), nrow = 3, byrow = T)
# y <- to_categoricalF(y = c(1, 2, 3), 3)
# shuffle_data(x, y)

##############################
#     Mini-batch split       #
##############################
batch_split <- function(x, y, batch_size){
  # batch_split
  # Args :
  #   --x: input x, shape: [m, in_dim]
  #   --y: input labels, one-hot, shape: [m, out_dim]
  #   --batch_size: batch size
  # return :
  #   --batches in list, [[1]] = batch, batch = [[x], [y]]
  batches <- list()
  num_examples <- nrow(x)
  num_batches <- floor(num_examples / batch_size)
  count <- 1
  for (i in 1 : num_batches){
    batch_x <- x[((i - 1) * batch_size + 1) : (i * batch_size),]
    batch_y <- y[((i - 1) * batch_size + 1) : (i * batch_size),]
    batch <- list(x = batch_x, y = batch_y)
    batches[[i]] <- batch
    count <- count + 1
  }

  if (num_examples %% batch_size != 0){
    batch_x <- x[((count - 1) * batch_size + 1) : nrow(x),]
    batch_y <- y[((count - 1) * batch_size + 1) : nrow(y),]
    batch <- list(x = batch_x, y = batch_y)
    batches[[count]] <- batch
  }
  return (batches)
}

# unit test
# x <- matrix(c(3, 4, 5, 6, 7, 8, 1, 2, 3, 10, 11, 12), nrow = 4, byrow = T)
# y <- to_categoricalF(y = c(1, 2, 3, 5), 5)
# batch_split(x, y, batch_size = 4)

##############################
#     validation split       #
##############################
validation_split <- function(x, y, split_rate = 0.1){
  # To do validation split
  # Args :
  #   --x: input data, shape: [m, in_dim]
  #   --y: one-hot labels, shape: [m, out_dim]
  #   --split_rate: VALIDATION split, default is 0.1
  # return :
  #   a list of split results [[train], [val]], train = [[x], [y]]
  # we need to map back y to original space
  y_ <- apply(y, 1, function(x) which.max(x))
  #indices <- createDataPartition(y_, p = (1 - split_rate), list = F)
  data_list <- shuffle_data(x, y)
  x <- data_list$x
  y <- data_list$y
  num_examples <- nrow(x)
  num_train <- floor((1. - split_rate) * num_examples)
  train_x <- x[1 : num_train, ]
  train_y <- y[1 : num_train, ]
  val_x <- x[(num_train + 1) : nrow(x), ]
  val_y <- y[(num_train + 1) : nrow(y), ]
  train <- list(x = train_x, y = train_y)
  val <- list(x = val_x, y = val_y)

  return (list(train = train, val = val))
}

# unit test
# x <- matrix(c(3, 4, 5, 6, 7, 8, 1, 2, 3, 10, 11, 12), nrow = 4, byrow = T)
# y <- to_categoricalF(y = c(1, 2, 3, 5), 5)
# validation_split(x, y, split_rate = 0.5)

##############################
#       Parameters map       #
##############################
# map_params <- function(models, meta_params){
#   # Map updated parameters back to model
#   # Args :
#   #   --models: model's instance
#   #   --meta_params: updated parameters
#   # return :
#   #   --models: updated model
#   meta_names <- names(models$model_ins)
#   for (name in meta_names){
#     models$model_ins[[name]]$dense_layer@W <- meta_params[[name]]$W
#     models$model_ins[[name]]$dense_layer@b <- meta_params[[name]]$b
#   }
#   return (models)
# }

##############################
#       Gradient checking    #
##############################
gradient_checking <- function(){ # here we check two layers
  x <- matrix(rnorm(20), nrow = 4, ncol = 5)
  y <- matrix(c(1, 0, 0, 1, 1, 0, 0, 1), nrow = 4, ncol = 2, byrow = T)
  models <- Model(dims = c(2, 2), input_shape = 5,
                  hidden_activation = 'relu',
                  out_activation = 'softmax',
                  kernel_initializer = 'random', l1 = 0.1, l2 = 0.1)
  loss_func <- Loss(method = 'celoss', l1 = 0.1, l2 = 0.1)
  meta_params <- Forward(models = models, x = x)
  # gradient checking
  W1 <- meta_params[['hidden1']]$W
  b1 <- meta_params[['hidden1']]$b
  W2 <- meta_params[['final_layer']]$W
  b2 <- meta_params[['final_layer']]$b
  # analytical gradients
  meta_grads <- Backward(meta_params, y, models)
  dW1 <- meta_grads[['hidden1']]$dW
  db1 <- meta_grads[['hidden1']]$db
  dW2 <- meta_grads[['final_layer']]$dW
  db2 <- meta_grads[['final_layer']]$db
  meta_grads <- c(c(dW1), c(db1), c(dW2), c(db2))

  dimW1 <- dim(W1)
  vecW1 <- c(W1)
  dimb1 <- dim(b1)
  vecb1 <- c(b1)
  dimW2 <- dim(W2)
  vecW2 <- c(W2)
  dimb2 <- dim(b2)
  vecb2 <- c(b2)
  # make them a vector
  params <- c(W1 = vecW1, b1 = vecb1, W2 = vecW2, b2 = vecb2)
  epsilon <- 1e-6
  num_grads <- c()

  for (i in 1 : length(params)){
    # numeric gradients
    params_le <- params
    params_le[i] <- params_le[i] + epsilon
    params_ri <- params
    params_ri[i] <- params_ri[i] - epsilon
    W1 <- matrix(params_le[1 : prod(dimW1)], nrow = dimW1[1])
    b1 <- matrix(params_le[(prod(dimW1) + 1) :  (prod(dimW1) + prod(dimb1))], nrow = dimb1[1])
    W2 <- matrix(params_le[(prod(dimW1) + 1 + prod(dimb1)) :  (prod(dimW1) + prod(dimb1) + prod(dimW2))], nrow = dimW2[1])
    b2 <- matrix(params_le[(prod(dimW1) + 1 + prod(dimb1) + prod(dimW2)) : length(params_le)], nrow = dimb2[1])
    meta_params2 <- list('hidden1' = list(W = W1, b = b1),
                         'final_layer' = list(W = W2, b = b2))
    meta_params_temp <- Forward(models = models, x = x, params = meta_params2)

    batch_cost1 <- loss_func(output = meta_params_temp$final_layer$x,
                             y = y, params = meta_params_temp)
    W1 <- matrix(params_ri[1 : prod(dimW1)], nrow = dimW1[1])
    b1 <- matrix(params_ri[(prod(dimW1) + 1) :  (prod(dimW1) + prod(dimb1))], nrow = dimb1[1])
    W2 <- matrix(params_ri[(prod(dimW1) + 1 + prod(dimb1)) :  (prod(dimW1) + prod(dimb1) + prod(dimW2))], nrow = dimW2[1])
    b2 <- matrix(params_ri[(prod(dimW1) + 1 + prod(dimb1) + prod(dimW2)) : length(params_le)], nrow = dimb2[1])
    meta_params2 <- list('hidden1' = list(W = W1, b = b1),
                         'final_layer' = list(W = W2, b = b2))
    meta_params_temp <- Forward(models = models, x = x, params = meta_params2)
    batch_cost2 <- loss_func(output = meta_params_temp$final_layer$x, params = meta_params_temp,
                             y = y)
    numeric_grad <- (batch_cost1 - batch_cost2) / (2 * epsilon)
    num_grads[i] <- numeric_grad
  }

  numerator <- sqrt(sum((num_grads - meta_grads) ** 2))
  denominator <- sqrt(sum(num_grads ** 2)) + sqrt(sum(meta_grads ** 2))

  diff <- round(numerator / denominator, 10)

  if (diff > epsilon){
    print(glue('Gradient checking failed, diff is {diff}!'))
  } else{
    print(glue('Gradient checking passed, diff is {diff}!'))
  }
}

# unit test for grad check
# gradient_checking()

##############################
#         Grad-CAM           #
##############################
saliency_map <- function(models, x, y, params, img_size) {
  # This function computes saliency map
  # https://arxiv.org/abs/1911.11293
  # Args :
  #   --models: models instance
  #   --x: input x
  #   --y: output label
  #   --params
  # return :
  #   --final: saliency map
  # 1. compute forward
  meta_params <- Forward(models = models, x = x, params = params)
  # 2. get logits of label
  #output <- meta_params$final_layer$z # logits
  #output <- output * y # shape [m, num_classes]
  # 3. compute gradient for output w.r.t. the inputs
  meta_grads <- Backward(meta_params, y = -1, models) # for saliency map's gradient
  # 4. get map
  saliency <- abs(meta_grads$hidden1$dA) # [m, in_dim]
  # 5. reshape to get image size
  img_channels <- dim(saliency)[2] / (img_size ** 2)
  final <- list()
  for (i in 1 : dim(saliency)[1]){
    s <- saliency[i, ] # [in_dim, ]
    dim(s) <- c(img_size, img_size, img_channels)
    temp <- matrix(rep(0., img_size ** 2), nrow = img_size)
    for (j in 1 : img_channels){
      s[,,j] <- t(s[,,j])
      temp <- pmax(temp, s[,,j])
    } # s has shape: [h, w, c], temp has shape [h, w]
    final[[i]] <- image(temp) # convert to image
  }
  # `final` is final output saliency map
  return (final)
}

##############################
#          Vis NN            #
##############################
generate_data <- function(no_nodes_per_layer = c(2, 3, 2, 3, 2)) {
  # from: https://github.com/brandonyph/MLP-Animation/blob/main/cutom%20MLP%20Plot.Rmd
  layers <- length(no_nodes_per_layer)
  if(length(no_nodes_per_layer) < layers){
    no_nodes_per_layer <- c(no_nodes_per_layer,
                            seq(1, layers - length((no_nodes_per_layer))))
  }
  df <- c()
  max_nodes <- max(no_nodes_per_layer)

  ##Creating layers
  for (i in 1 : layers) {
    df$layers <- c(df$layers, rep(i, no_nodes_per_layer[i]))
  }

  ##Creating nodes
  for (i in 1 : layers) {
    nodes_no <- no_nodes_per_layer[i]
    diff <- (max_nodes - nodes_no)/2
    df$nodes <- c(df$nodes, seq(1, nodes_no) + diff)
  }
  df$sizes <- rep(0, length(df$layers))
  df <- data.frame(df)

  return(df)
}

stack_overall <- function(Overall, layers, df){
  # from: https://github.com/brandonyph/MLP-Animation/blob/main/cutom%20MLP%20Plot.Rmd
  df2 <- df
  normalize <- function(x)
  {
    return((x - min(x)) /(max(x) - min(x)))
  }
  for (l in 1 : layers) {
      df2$sizes[df2$layers == (l + 1)] <- Overall$bias[[l]]
  }
  # Increase size for better plotting
  #df2$sizes <- normalize(df2$sizes)

  return (df2)
}

vis_nn <- function(no_nodes_per_layer = c(2, 3, 2, 3, 2),
                   Overall, layers, cur, oldOverall){
  # from: https://github.com/brandonyph/MLP-Animation/blob/main/cutom%20MLP%20Plot.Rmd
  df <- generate_data(no_nodes_per_layer)
  df2 <- stack_overall(Overall, layers, df)
  df4 <- df2
  ## Create base Canvas
  png(paste(c("./www/plotNN_", cur, ".png"), collapse = ''))
  p <-
    ggplot(data = df4) + geom_point(aes(x = layers, y = nodes, size = sizes * 20,
                                        color = "red")) +
    theme_bw()

  ##function to add all arrows
  for (i in 1 : (layers)) {
    for (j in 1 : (no_nodes_per_layer[i])) {
      for (o in 1 : (no_nodes_per_layer[i + 1])) {
        x1 <- i
        y1 <- df4[df4$layers == i, ]$nodes[j]
        x2 <- i + 1
        y2 <- df4[df4$layers == (i + 1), ]$nodes[o]

        coor <- data.frame(x = c(x1, x2), y = c(y1, y2))
        weights <- Overall$weight[[x1]][[j, o]]
        if (is.null(oldOverall))
          weights_old <- weights
        else
          weights_old <- oldOverall$weight[[x1]][[j, o]]

        p <- p + geom_path(data = coor, aes(x = x, y = y), color = ifelse(weights > weights_old, 'green', 'blue'), size = 0.5,
                           alpha = ifelse(weights > weights_old, 10, 0.5))
      }
    }
  }
  p <- p +
    theme(legend.title = element_blank(), axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          panel.border = element_blank(),
          panel.grid.major = element_blank(),
          axis.title.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.title.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid.minor = element_blank())
  print(p)
  dev.off()
  return (Overall)
}

# setwd('/Users/kunhongyu/Desktop/DSBA/Year1/Semester2/Advanced R/project/code')
# library(imager)
# x <- load.image("img.png")
# x <- grayscale(x, method = "Luma", drop = TRUE)
# dim(x)
# x <- resize(x, 28, 28)
# x <- x[, , 1, 1]
# dim(x) <- c(1, 784)
# # image(x)
#
# dims <- c(2,3,3)
# y <- c(1,2,2)
# y <- to_categoricalF(y, 2)
# img_size <- 28
# models <- Model(dims = dims, input_shape = 784, hidden_activation = 'relu',
#                 out_activation = 'softmax')
# final <- saliency_map(models, x, y, params = NULL, img_size)

