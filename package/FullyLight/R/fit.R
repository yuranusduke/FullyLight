# This file is used to train the model like Keras `fit` function
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/04

library(glue)
library(progress)
library(factoextra)
library(ggimage)

rotate <- function(x, data_name){
  # rotate input image
  f <- list.files(paste(c('./show/', data_name), collapse = ''), include.dirs = F, full.names = T, recursive = F)
  file.remove(f)
  dim(x) <- c(nrow(x), 28, 28)
  for (i in 1 : dim(x)[1]){
    x[i,,] <- t(x[i,,])
    temp <- t(apply(x[i,28:1,], 2, rev))
    png::writePNG(temp, paste(c("./show/", data_name, '/', i, '.png'), collapse = ''))
  }
}

setGeneric(name = 'fit',
           def = function(object, ...){
             standardGeneric('fit')
           })
setMethod(f = 'fit',
          signature = 'FullyModel',
          definition = function(object, x, y, models, compile_funcs,
                                validation_rate = 0.0,
                                validation_data = NULL, epochs = 20,
                                batch_size = 32, verbose = 1,
                                shuffle = T, isimage = F, data_name = NULL, output = F){
            # This function trains model using input x and y
            # Args :
            #   --x: input x, shape : [m, dim]
            #   --y: input y ont-hot, shape : [m, num_classes]
            #   --models: from 'Model' instance
            #   --validation_rate: split rate for validation
            #   --validation_data: validation data, in tuple list(x =, y = )
            #   --epochs: training epochs, default is 20
            #   --batch_size: mini-batch size, default is 32
            #   --verbose: 1 as default for printing training history, 0 does not print
            #   --shuffle: default is T shuffling data before each epoch
            #   --isimage: default is False
            #   --data_name
            #   --output: shiny output?
            # return :
            #   --models: trained models
            #   --history: training history, containing loss and acc
            #   --meta_params
            # 1. compile first
            loss_func <- compile_funcs$loss_func
            optimizer_func <- compile_funcs$optimizer_func
            metric_func <- compile_funcs$metric_func
            meta_params <- NULL
            m <- NULL
            v <- NULL
            losses <- c()
            accs <- c()
            # 2. start training
            print(glue("Start training now..."))
            count <- 0
            steps <- 0
            oldOverall <- NULL
            # we randomly get 200 samples for visualization
            indices <- sample(1 : nrow(x), size = 350, replace = T)
            samples_x <- x[indices, ]
            samples_y <- y[indices, ]
            x <- samples_x
            y <- samples_y
            rand_color <- runif(1)
            samples_y <- apply(samples_y, 1, function(x) which.max(x))
            if (isimage & output)
              rotate(samples_x, data_name) # save image
            for (epoch in 1 : epochs){ # 2.1 iterate each epoch
                epoch_cost <- 0.
                epoch_acc <- 0.
                if (shuffle) { # shuffle first
                  shuffled_data <- shuffle_data(x = x, y = y)
                  x <- shuffled_data$x
                  y <- shuffled_data$y
                }
                if (validation_rate > 0.0){ # validation split
                  partitioned_data <- validation_split(x = x, y = y, split_rate = validation_rate)
                  train <- partitioned_data$train
                  train_x <- train$x
                  train_y <- train$y
                  val <- partitioned_data$val
                  val_x <- val$x
                  val_y <- val$y
                } else if (!is.null(validation_data)){
                  train_x <- x
                  train_y <- y
                  val_x <- validation_data$x
                  val_y <- validation_data$y
                } else {
                  train_x <- x
                  train_y <- y
                  val_x <- NULL # flag
                }

              # then, create batches of inputs
              batches <- batch_split(x = train_x, y = train_y, batch_size = batch_size)
              # Define printing bar
              if (!is.null(val_x))
                pb <- progress_bar$new(format = glue("Epoch {epoch} / {epochs} [:bar] :percent || [TRAIN cost :cost metric :acc%] [VAL cost :val_cost metric :val_acc%]."),
                  clear = FALSE, total = length(batches), width = 120)
              else
                pb <- progress_bar$new(format = glue("Epoch {epoch} / {epochs} [:bar] :percent || [TRAIN cost :cost metric :acc%]."),
                                       clear = FALSE, total = length(batches), width = 120)
              # 2.2 iterate each batch
              for (batch in batches){
                batch_x <- batch$x
                batch_y <- batch$y
                # 2.2.1 forward pass
                meta_params <- Forward(models = models, x = batch_x, params = meta_params)
                # 2.2.2 compute cost
                batch_cost <- loss_func(output = meta_params$final_layer$x,
                                        y = batch_y, params = meta_params)
                # 2.2.3 backward pass to get gradients
                meta_grads <- Backward(meta_params = meta_params, y = batch_y, models = models)
                # 2.2.4 update parameters
                updated_info <- optimizer_func(meta_grads = meta_grads,
                                               meta_params = meta_params, m = m, v = v)
                if ('m' %in% names(updated_info)){
                  m <- updated_info$m
                }
                if ('v' %in% names(updated_info)){
                  v <- updated_info$v
                }
                # 2.2.5 put back parameters into the model
                meta_params <- updated_info$meta_params

                # 2.2.6 get metric
                # print(meta_params$final_layer$x_prev)
                batch_metric <- metric_func(output = meta_params$final_layer$x, y = batch_y)
                if (count %% 2 == 0){
                  steps <- steps + 1
                  losses <- c(losses, batch_cost)
                  accs <- c(accs, batch_metric)
                  log_data <- data.frame(loss = losses, acc = accs)
                  # loss
                  if (output){ # shiny output
                    png(paste(c("./www/his_", count, ".png"), collapse = ''))
                    his <- ggplot(data = log_data) + geom_line(aes(x = 1 : steps, y = loss, color = 'Loss'), size = 2) +
                      geom_line(aes(x = 1 : steps, y = acc, color = 'Metric'), size = 2) + xlab('Iterations') + ylab('Values') +
                      scale_colour_manual("", breaks = c("Loss", "Metric"),
                                          values = c("brown", "darkolivegreen"))
                    print(his)
                    dev.off()

                    # classification result
                    meta_params_samples <- Forward(models = models, x = samples_x, params = meta_params)
                    final_x <- meta_params_samples[['final_layer']]$x

                    # vis for neural net
                    weight <- list()
                    bias <- list()
                    model_stat <- list()
                    Overall <- list()
                    meta_names <- names(meta_params_samples)
                    countl <- 1
                    first_layer_dim <- dim(meta_params_samples[['hidden1']]$x_prev)[2]
                    first_layer_dim <- ifelse(first_layer_dim > 10, 10, first_layer_dim)
                    dims <- c(first_layer_dim)
                    for (name in meta_names){
                      real <- F
                      if(dim(meta_params[[name]]$W)[2] > 10){
                        ver_coor <- sample(1 : dim(meta_params[[name]]$W)[2], size = 10, replace = F)
                        weight[[countl]] <- as.matrix(t(meta_params[[name]]$W[, ver_coor, drop = F])) # only visualize 10 for faster computation
                        real <- T
                      }
                      if (dim(meta_params[[name]]$W)[1] > 10)
                      {
                        if (real){
                          ver_coor <- sample(1 : dim(meta_params[[name]]$W)[2], size = 10, replace = F)
                          hor_coor <- sample(1 : dim(meta_params[[name]]$W)[1], size = 10, replace = F)
                          weight[[countl]] <- as.matrix(t(meta_params[[name]]$W[hor_coor, ver_coor, drop = F]))
                        }
                        else{
                          hor_coor <- sample(1 : dim(meta_params[[name]]$W)[1], size = 10, replace = F)
                          weight[[countl]] <- as.matrix(t(meta_params[[name]]$W[hor_coor, , drop = F]))
                        }
                      }
                      else if (dim(meta_params[[name]]$W)[2] <= 10 & dim(meta_params[[name]]$W)[1] <= 10)
                        weight[[countl]] <- as.matrix(t(meta_params[[name]]$W))
                      if(dim(meta_params[[name]]$b)[2] > 10)
                        bias[[countl]] <- as.matrix(t(meta_params[[name]]$b[,1 : 10, drop = F])) # keep dimension
                      else
                        bias[[countl]] <- as.matrix(t(meta_params[[name]]$b))
                      dims <- c(dims, dim(bias[[countl]])[1])
                      countl <- countl + 1
                    }
                    model_stat$weight <- weight
                    model_stat$bias <- bias

                    oldOverall <- vis_nn(no_nodes_per_layer = dims,
                           Overall = model_stat,
                           layers = length(meta_names), cur = count,
                           oldOverall = oldOverall)

                    # DR
                    png(paste(c("./www/clas_", count, ".png"), collapse = ''))
                    res.pca <- prcomp(final_x, scale = TRUE)
                    final_x <- res.pca$x[,1 : 2]
                    final_x <- data.frame(final_x)
                    if (!isimage){ # not image
                      clas <- ggplot(final_x, aes(x = PC1, y = PC2)) +
                        geom_point(aes(color = as.factor(samples_y)), size = 5) + xlab('') + ylab('') +
                        theme(legend.title = element_blank(), axis.text.x = element_blank(),
                              axis.text.y = element_blank(),
                              panel.border = element_blank(),
                              panel.grid.major = element_blank(),
                              axis.title.x = element_blank(),
                              axis.ticks.x = element_blank(),
                              axis.title.y = element_blank(),
                              axis.ticks.y = element_blank(),
                              panel.grid.minor = element_blank()) + scale_color_manual(values = rainbow(length(unique(as.factor(samples_y))), start = rand_color))
                    }
                    else{ # image
                      all_images <- list.files(paste(c('./show/', data_name), collapse = ''), pattern = "*.png$",
                                                    include.dirs = F, full.names = T, recursive = F)
                      all_images <- all_images[1 : length(samples_y)]
                      all_images <- all_images[order(as.numeric(gsub("[^0-9]+", "", all_images)),
                                                         decreasing = F)] # sorting is important!!

                      final_x$images <- all_images

                      #samples_y_ <- sapply(all_images, function(x) unlist(strsplit(unlist(strsplit(x, '_'))[2], '\\.'))[1])
                      #samples_y_ <- sapply(samples_y_, as.numeric)
                      clas <- ggplot(final_x, aes(x = PC1, y = PC2)) +
                        geom_point(aes(color = as.factor(samples_y)), size = 10) + xlab('') + ylab('') +
                        theme(legend.title = element_blank(), axis.text.x = element_blank(),
                              axis.text.y = element_blank(),
                              panel.border = element_blank(),
                              panel.grid.major = element_blank(),
                              axis.title.x = element_blank(),
                              axis.ticks.x = element_blank(),
                              axis.title.y = element_blank(),
                              axis.ticks.y = element_blank(),
                              panel.grid.minor = element_blank()) + geom_image(aes(image = images), size = .03)
                    }
                    print(clas)
                    dev.off()
                  }
                }
                if (verbose == 1) {
                  Sys.sleep(0.2)
                  if (is.null(val_x))
                    pb$tick(tokens = list(cost = round(batch_cost, 3),
                                 acc = round(100 * batch_metric, 2)))
                  else {
                    eval_res <- Evaluate(x = val_x, y = val_y, models = models, params = meta_params,
                                         metric = 'acc')
                    # print(list(cost = round(batch_cost, 3),
                    #            acc = round(100 * batch_metric, 2),
                    #            val_cost = round(eval_res$loss, 3),
                    #            val_acc = round(100 * eval_res$eval_res, 2)))
                    pb$tick(tokens = list(cost = round(ifelse(!is.na(batch_cost) & !is.nan(batch_cost), batch_cost, 1e10), 3),
                                          acc = round(100 * ifelse(!is.na(batch_metric) & !is.nan(batch_metric), batch_metric, 0), 2),
                                          val_cost = round(ifelse(!is.na(eval_res$loss) & !is.nan(eval_res$loss), eval_res$loss, 1e10), 3),
                                          val_acc = round(100 * ifelse(!is.na(eval_res$eval_res) & !is.nan(eval_res$eval_res), eval_res$eval_res, 0), 2)))
                  }
                }
                count <- count + 1
                epoch_cost <- epoch_cost + batch_cost
                epoch_acc <- epoch_acc + batch_metric
              }
              epoch_cost <- epoch_cost / count
              epoch_acc <- epoch_acc / count
            }

            history <- list(loss = losses, acc = accs)
            return (list(models = models, history = history, params = meta_params))
            }
          )


#' Function to train FullyLight model instance
#'
#' Function will call engine of all inside functions to start training FullyLight model,
#' one may put return into \code{FullyLight::Evaluate} function.
#'
#' @author Kunhong Yu
#' @param x matrix. Input design matrix, shape: m, input_dim.
#' @param y matrix. One-hot form of labels, shape: m, num_classes.
#' @param models 'Model' instance.
#' @param compile_funcs All compiled functions from \code{FullyLight::Compile}.
#' @param validation_rate numeric. Validation split, default is 0.0.
#' @param validation_data list. Validation data, in tuple list(x =, y = ), default is NULL.
#' @param epochs numeric. Training epochs, default is 20.
#' @param batch_size numeric. Training batch size, default is 32.
#' @param verbose numeric. 1 as default for printing training history, 0 does not print.
#' @param shuffle bool. Default is TRUE shuffling data before each epoch.
#' @param isimage bool. Default is False for not using image as data.
#' @param data_name character string. Data set name, default is NULL.
#' @param output bool. Default is FALSE, TRUE for shiny output.
#' @return History of learning, including training loss and metric for each learning step and trained model, etc..
#' @import ggplot2
#' @import methods
#' @import grDevices
#' @import png
#' @import graphics
#' @examples
#' x <- matrix(c(3, 7, 9, 10, 0, 9), nrow = 2)
#' y <- matrix(c(0, 1, 0, 1, 0, 0), nrow = 2, byrow = TRUE)
#' models <- FullyLight::Model(dims = c(3, 10, 3),
#'                             input_shape = 3,
#'                             hidden_activation = 'relu',
#'                             out_activation = 'softmax',
#'                             kernel_initializer = 'random',
#'                             l1 = 0,
#'                             l2 = 0)
#' compile_funcs <- FullyLight::Compile(loss = 'categorical_crossentropy',
#'                                      optimizer = 'adam',
#'                                      metric = 'f1',
#'                                      learning_rate = 1e-3,
#'                                      l1 = 0,
#'                                      l2 = 0)
#' res <- Fit(x = x,
#'            y = y,
#'            models = models,
#'            compile_funcs = compile_funcs,
#'            validation_rate = 0.01,
#'            validation_data = NULL,
#'            epochs = 1,
#'            batch_size = 32,
#'            verbose = 1,
#'            shuffle = TRUE,
#'            isimage = FALSE,
#'            data_name = 'iris')
#' @export
Fit <- function(x, y, models, compile_funcs,
                validation_rate = 0.0,
                validation_data = NULL, epochs = 20,
                batch_size = 32, verbose = 1,
                shuffle = T, isimage = F, data_name = NULL,
                output = F){
  # This function is used to fit the model
  # Args :
  #   --x: input x, shape : [m, dim]
  #   --y: input y ont-hot, shape : [m, num_classes]
  #   --models: from 'Model' instance
  #   --validation_rate: split rate for validation
  #   --validation_data: validation data, in tuple list(x =, y = )
  #   --epochs: training epochs, default is 20
  #   --batch_size: mini-batch size, default is 32
  #   --verbose: 1 as default for printing training history, 0 does not print
  #   --shuffle: default is T shuffling data before each epoch
  #   --isimage: default is False
  #   --data_name
  fit_func <- selectMethod('fit', signature = 'FullyModel')
  res <- fit_func(x = x, y = y, models = models, compile_funcs = compile_funcs,
           validation_rate = validation_rate,
           validation_data = validation_data, epochs = epochs,
           batch_size = batch_size, verbose = verbose,
           shuffle = shuffle, isimage = isimage, data_name = data_name, output = output)

  return (res)
}
