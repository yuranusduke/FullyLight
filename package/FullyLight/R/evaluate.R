# This file contains operations about evaluating models, including prediction
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/04

setGeneric(name = 'evaluate',
           def = function(object, ...){
             standardGeneric('evaluate')
           })
setMethod(f = 'evaluate',
          signature = 'FullyModel',
          definition = function(object, x, y, models, params,
                                loss = 'categorical_crossentropy',
                                metric = 'acc'){
            # This function is used to evaluate model
            # Args :
            #   --x: input data, shape: [m, in_dim]
            #   --y: one-hot labels, shape: [m, out_dim]
            #   --models: from 'Mode' instance
            #   --params: parameters
            #   --loss: default is 'categorical_crossentropy'
            #   --metric: default is 'acc'
            # return :
            #   --eval_res, loss
            funcs <- Compile(loss = loss, metric = metric)
            loss_func <- funcs$loss_func
            metric_func <- funcs$metric_func
            pred_func <- selectMethod('predict', signature = 'FullyModel')
            preds <- pred_func(x = x, models = models, params = params, type = 'probs')
            loss <- loss_func(output = preds, y = y)
            #print(loss)
            eval_res <- metric_func(output = preds, y = y)

            return (list(eval_res = eval_res, loss = loss))
          }
)

setGeneric(name = 'predict',
           def = function(object, ...){
             standardGeneric('predict')
           })
setMethod(f = 'predict',
          signature = 'FullyModel',
          definition = function(object, x, models, params, type = 'probs'){
            # predict using trained model
            # Args :
            #   --x: input data, shape: [m, in_dim]
            #   --models: from 'Model' instance
            #   --type: 'probs'/'class', default is 'probs'
            #   --params: parameters
            # return :
            #   --preds
            assert_that(type %in% c('probs', 'class'))
            res <- Forward(models = models, x = x, params = params)
            softmax_preds <- res$final_layer$x
            if (type == 'probs') return (softmax_preds)
            else {
              preds <- apply(softmax_preds, 1, function(x) which.max(x))
              return (preds)
            }
          }
)

#' Function to evaluate FullyLight model instance
#'
#' Function will call engine of all inside functions to evaluate FullyLight model.
#'
#' @author Kunhong Yu
#' @param x matrix. Input design matrix, shape: m, input_dim.
#' @param y matrix. One-hot form of labels, shape: m, num_classes.
#' @param models 'Model' instance. If we have return from \code{FullyLight::Fit} as \code{res}, this parameter should be \code{res$models}.
#' @param params list. If we have return from \code{FullyLight::Fit} as \code{res}, this parameter should be \code{res$params}.
#' @param loss character string. Only support 'categorical_crossentropy', don't modify!
#' @param metric character string. Metric for classification, 'acc'/'recall'/'f1'/'precision', default is 'acc'.
#' @return A list of results, including \code{eval_res} as metric on test set, \code{loss} for loss on test set.
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
#' res <- FullyLight::Fit(x = x,
#'                        y = y,
#'                        models = models,
#'                        compile_funcs = compile_funcs,
#'                        validation_rate = 0.01,
#'                        validation_data = NULL,
#'                        epochs = 1,
#'                        batch_size = 32,
#'                        verbose = 1,
#'                        shuffle = TRUE,
#'                        isimage = FALSE,
#'                        data_name = 'iris')
#' eval_res <- Evaluate(x = x, y = y, models = res$models, params = res$params,
#'                      loss = 'categorical_crossentropy', metric = 'f1')
#' @export
Evaluate <- function(x, y, models, params,
                     loss = 'categorical_crossentropy',
                     metric = 'acc'){
  # This function is used to evaluate model
  # Args :
  #   --x: input data, shape: [m, in_dim]
  #   --y: one-hot labels, shape: [m, out_dim]
  #   --models: from 'Mode' instance
  #   --params: parameters
  #   --loss: default is 'categorical_crossentropy'
  #   --metric: default is 'acc'
  # return :
  #   --eval_res, loss
  evaluate_func <- selectMethod('evaluate', signature = 'FullyModel')
  res <- evaluate_func(x = x, y = y, models = models, params = params,
                       loss = loss,
                       metric = metric)
  return (res)
}

#' Function to make a prediction with trained FullyLight model
#'
#' Function will call engine of all inside functions to make a prediction using pre-trained FullyLight model.
#'
#' @author Kunhong Yu
#' @param x matrix. Input design matrix, shape: m, input_dim.
#' @param models 'Model' instance. If we have return from \code{FullyLight::Fit} as \code{res}, this parameter should be \code{res$models}.
#' @param params list. If we have return from \code{FullyLight::Fit} as \code{res}, this parameter should be \code{res$params}.
#' @param type character string. 'probs' or 'class', default is 'probs', in line with simple R \code{predict}.
#' @return Predictions, shape: m, .
#' @import assertthat
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
#' res <- FullyLight::Fit(x = x,
#'                        y = y,
#'                        models = models,
#'                        compile_funcs = compile_funcs,
#'                        validation_rate = 0.01,
#'                        validation_data = NULL,
#'                        epochs = 1,
#'                        batch_size = 32,
#'                        verbose = 1,
#'                        shuffle = TRUE,
#'                        isimage = FALSE,
#'                        data_name = 'iris')
#' preds <- Predict(x = x, models = res$models, params = res$params, type = 'probs')
#' @export
Predict <- function(x, models, params, type = 'probs'){
  # predict using trained model
  # Args :
  #   --x: input data, shape: [m, in_dim]
  #   --models: from 'Model' instance
  #   --type: 'probs'/'class', default is 'probs'
  #   --params: parameters
  # return :
  #   --preds
  predict_func <- selectMethod('predict', signature = 'FullyModel')
  preds <- predict_func(x = x, models = models, params = params, type = type)

  return (preds)
}
