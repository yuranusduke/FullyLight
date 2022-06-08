# This file is used to compile the model like Keras `compile` function
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/03

setGeneric(name = 'compile',
           def = function(object, ...){
             standardGeneric('compile')
           })
setMethod(f = 'compile',
          signature = 'FullyModel',
          definition = function(object, loss = 'categorical_crossentropy',
                                optimizer = 'adam', metric = 'acc', learning_rate = 1e-3, l1 = 0., l2 = 0.,
                                ...){
            # In this project, we simply define our loss as 'categorical_crossentropy'
            # since we only do multi-class classification, meanwhile, we specify 1e-2 as start learning rate
            # for adaptive learning rate algorithm
            # Args :
            #   --optimizer: 'adam'/'momentum'/'sgd'/'rmsprop', default is 'adam'
            #   --metric: 'acc'/'recall'/'f1'/'precision', default is 'acc'
            #   --l1/l2/learning rate
            # return :
            #   --optimizer_func/loss_func/metric_func
            optimizer_func <- Optimizer(method = optimizer, learning_rate = learning_rate)
            if (loss == 'categorical_crossentropy')
              loss_func <- Loss(method = 'celoss', l1 = l1, l2 = l2)
            metric_func <- Metrics(method = metric)

            return (list(optimizer_func = optimizer_func, loss_func = loss_func, metric_func = metric_func))
          })


#' Function to compile FullyLight model instance
#'
#' Function will call engine of all inside functions to compile FullyLight model,
#' one may put return into \code{FullyLight::Fit} function.
#'
#' @author Kunhong Yu
#' @param loss character string. Only support 'categorical_crossentropy', don't modify!
#' @param optimizer character string. 'adam' is default, also supports 'sgd'/'rmsprop'/'momentum'.
#' @param metric character string. Metric for classification, 'acc'/'recall'/'f1'/'precision', default is 'acc'.
#' @param learning_rate numeric. Learning rate, default is 1e-3.
#' @param l1 numeric. L1 regularization penalty.
#' @param l2 numeric. L2 regularization penalty.
#' @param ... All other arguments related to the function.
#' @return A list of compiled functions, including optimizer, loss function and metrics.
#' @import stats
#' @examples
#' compile_funcs <- Compile(loss = 'categorical_crossentropy',
#'                          optimizer = 'adam',
#'                          metric = 'f1',
#'                          learning_rate = 1e-3,
#'                          l1 = 0,
#'                          l2 = 0)
#' @export
Compile <- function(loss = 'categorical_crossentropy',
                    optimizer = 'adam', metric = 'acc', learning_rate = 1e-3, l1 = 0., l2 = 0., ...){
  # Define Compile function belonging to `Model` class
  # Args :
  #   --optimizer: 'adam'/'momentum'/'sgd'/'rmsprop', default is 'adam'
  #   --metric: 'acc'/'recall'/'f1'/'precision', default is 'acc'
  #   --l1/l2/learning rate
  #
  # return :
  #   --optimizer_func/loss_func/metric_func
  compile_func <- selectMethod('compile', signature = 'FullyModel')
  res <- compile_func(loss = loss, optimizer = optimizer, metric = metric,
                      learning_rate = learning_rate, l1 = l1, l2 = l2, ...)

  return (res)
}

