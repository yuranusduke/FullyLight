# This script builds whole Fully Connected model
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/03

#' Class S4 representing a FullyLight Model.
#'
#' Class defines an object with FullyLight Model's characteristics.
#'
#' @name FullyModel-class
#' @slot dims Dimensions of NN
#' @slot input_shape Input's shape
#' @slot hidden_activation Hidden layer's activation
#' @slot out_activation Output activation
#' @slot kernel_initializer Kernel's initialization
#' @slot l1 L1 regularization
#' @slot l2 L2 regularization
#' @rdname FullyModel-class
#' @exportClass FullyModel
setClass('FullyModel',
         slots = list(dims = 'numeric',
                      input_shape = 'numeric',
                      hidden_activation = 'character',
                      out_activation = 'character',
                      kernel_initializer = 'character',
                      l1 = 'numeric',
                      l2 = 'numeric'),
         validity = function(object){
           if (!is.numeric(object@dims)) return ('Hidden dimension must be numeric!')
           if (!is.numeric(object@input_shape)) return ('Input shape must be numeric!')
           if (!is.character(object@hidden_activation)) return ('Hidden activation function must be character,
                                                                relu/sigmoid/tanh/prelu are valid!')
           if (!is.character(object@out_activation)) return ("Out activation function must be character,
                                                             linear/softmax are valid!")
           if (!is.character(object@kernel_initializer)) return ("kernel initializer must be character,
                                                                   random/xavier/ones/zero are valid!")
           if (!is.numeric(object@l1)) return ('L1 parameter must be numeric!')
           if (!is.numeric(object@l2)) return ('L2 parameter must be numeric!')

           return (T)
         })

##################################
#       Define full model        #
##################################
setGeneric(name = 'FullyModelDef',
           def = function(object, ...){
             standardGeneric('FullyModelDef')
           })
setMethod(f = 'FullyModelDef',
          signature = 'FullyModel',
          definition = function(object){
            # This function builds model for multiple layers
            # return :
            #   --models: models we obtain
            models <- list()
            in_dim <- object@input_shape
            count <- 1
            for (dim in object@dims[1 : (length(object@dims) - 1)]){
              layer <- Dense(in_dim = in_dim, out_dim = dim, activation = object@hidden_activation,
                             kernel_initializer = object@kernel_initializer, l1 = object@l1, l2 = object@l2)
              in_dim <- dim
              models[[paste0('hidden', count)]] <- layer
              count <- count + 1
            }
            final_layer <- Dense(in_dim = in_dim, out_dim = object@dims[length(object@dims)],
                                 activation = object@out_activation,
                                 kernel_initializer = object@kernel_initializer, l1 = object@l1, l2 = object@l2)
            models$`final_layer` <- final_layer
            return (models)
          })

##################################
#     Define forward pass        #
##################################
setGeneric(name = 'ForwardProp',
           def = function(object, ...){
             standardGeneric('ForwardProp')
           })
setMethod(f = 'ForwardProp',
          signature = 'FullyModel',
          definition = function(object, models, x, params = NULL){
            # Whole forward pass
            # Args :
            #   --models: all dense layer instance for each layer
            #   --x: input data
            # return :
            #   --meta_params: all layers' parameters reserved for backward pass
            meta_params <- list()
            for (count in 1 : (length(models) - 1)){
              layer <- models[[paste0('hidden', count)]]
              if (!is.null(params)){
                param <- params[[paste0('hidden', count)]]
                param <- list(W = param$W, b = param$b)
              } else param <- NULL
              res <- layer$forward(x, param) # res contains x_prev, W, b, z, x, ac_method
              meta_params[[paste0('hidden', count)]] <- res
              count <- count + 1
              x <- res$x
            }
            last_layer <- models$`final_layer`
            if (!is.null(params)){
              param <- params[['final_layer']]
              param <- list(W = param$W, b = param$b)
            } else param <- NULL
            res <- last_layer$forward(x, param) # res contains x_prev, W, b, z, x, ac_method
            meta_params$`final_layer` <- res
            return (meta_params)
          }
)

##################################
#     Define backward pass       #
##################################
setGeneric(name = 'BackwardProp',
           def = function(object, ...){
             standardGeneric('BackwardProp')
           })
setMethod(f = 'BackwardProp',
          signature = 'FullyModel',
          definition = function(object, y, meta_params, models){
            # Whole backward pass
            # Args :
            #   --y: one-hot encoding labels
            #   --meta_params: meta parameters
            #   --models: model's instance
            # return :
            #   --meta_grads: gradients w.r.t. all parameters in the model in current iteration
            # res contains x_prev, W, b, z, x, ac_method
            res <- meta_params[['final_layer']]
            meta_grads <- list()
            # final layer -- softmax
            dA <- 1.
            layer_backward_res <- Dense_Backward(dense_layer = models[['final_layer']]$dense_layer,
                                                 x_prev = res$x_prev,
                                                 x_cur = res$x, dA = dA,
                                                 W = res$W,
                                                 activation_method = res$ac_method, y = y)
            meta_grads[['final_layer']] <- layer_backward_res
            dA <- layer_backward_res$dA
            for (count in seq((length(meta_params) - 1), 1, by = -1)){ # go backward
              res <- meta_params[[paste0('hidden', count)]]
              layer_backward_res <- Dense_Backward(dense_layer = models[[paste0('hidden', count)]]$dense_layer,
                                                   x_prev = res$x_prev,
                                                   x_cur = res$x, dA = dA,
                                                   W = res$W,
                                                   activation_method = res$ac_method)
              meta_grads[[paste0('hidden', count)]] <- layer_backward_res
              dA <- layer_backward_res$dA # from previous layer
            }
            return (meta_grads)
          }
)

#' Function to build FullyLight model instance
#'
#' Function will call engine of all inside functions to complete model building using closure,
#' one may put return into \code{FullyLight::Fit} function.
#'
#' @author Kunhong Yu
#' @param dims a vector of numeric numbers. All layers' dims, excluding input layer.
#' @param input_shape numeric. Shape of input, should have 784 for MNIST and FashionMNIST.
#' @param hidden_activation character string. Hidden layer's activation, default is 'relu', one may also choose 'sigmoid'/'tanh'/'prelu'/'linear'.
#' @param out_activation character string. Output activation, default is 'softmax'.
#' @param kernel_initializer character string. Default is 'xavier', one can also choose 'random'.
#' @param l1 numeric. L1 regularization penalty.
#' @param l2 numeric. L2 regularization penalty.
#' @return A list of instances. \code{model_ins}: model instance and \code{fully_model}: model's definition.
#' @import glue
#' @import progress
#' @import factoextra
#' @import ggimage
#' @import caret
#' @examples
#' dims <- c(20,10,10)
#' input_shape <- 784
#' hidden_activation <- 'relu'
#' kernel_initializer <- 'xavier'
#' l1 <- 0.01
#' l2 <- 0.01
#' models <- Model(dims = dims,
#'                 input_shape = input_shape,
#'                 hidden_activation = hidden_activation,
#'                 out_activation = 'softmax',
#'                 kernel_initializer = kernel_initializer,
#'                 l1 = l1,
#'                 l2 = l2)
#' @export
Model <- function(dims, input_shape,
                  hidden_activation = 'relu',
                  out_activation = 'softmax',
                  kernel_initializer = 'xavier',
                  l1 = 0., l2 = 0.){
  # Get model
  # Args :
  #   --dims: all layers' dims, excluding input layer
  #   --input_shape: shape of input, should have 784 for MNIST
  #   --hidden_activation: hidden layer's activation, default is 'relu'
  #   --out_activation: output activation, default is 'softmax'
  #   --kernel_initializer: default is 'xavier'
  #   --l1/l2: defaults are 0
  # return :
  #   --model_ins: model instance
  fullymodel <- new(Class = 'FullyModel', dims = dims,
                    input_shape = input_shape,
                    hidden_activation = hidden_activation,
                    out_activation = out_activation,
                    kernel_initializer = kernel_initializer, l1 = l1, l2 = l2)
  model_ins <- FullyModelDef(fullymodel)
  return (list(model_ins = model_ins, fullymodel = fullymodel))
}

Forward <- function(models, x, params = NULL){
  # This function is used to run whole forward pass for our model
  # Args :
  #   --x: input data
  #   --models: model instance
  #   --params: default is NULL
  # return :
  #   --results
  ForwardProp <- selectMethod('ForwardProp', signature = 'FullyModel')
  results <- ForwardProp(models = models$model_ins, x = x, params = params)

  return (results)
}

Backward <- function(meta_params, y, models){
  # This function is used to run whole backward pass for our model
  # Args :
  #   --y: one-hot encoding labels
  #   --meta_params: meta parameters
  #   --models: model's instance
  # return :
  #   --meta_grads: gradients w.r.t. all parameters in the model in current iteration
  BackProp <- selectMethod('BackwardProp', signature = 'FullyModel')
  meta_grads <- BackProp(y = y, meta_params = meta_params, models = models$model_ins)

  return (meta_grads)
}

# unit test
# models <- Model(dims = c(2, 2), input_shape = 6,
#                 hidden_activation = 'relu', out_activation = 'softmax',
#                 kernel_initializer = 'ones')
# x <- matrix(c(10, 20 ,30, 40, 5, 0, 10, 20, 30, 40, 5, 0), nrow = 2)
# meta_params <- Forward(models, x)
# meta_params
# #
# y <- matrix(c(0, 1, 1, 0), nrow = 2, byrow = TRUE)
# # y # one-hot
# meta_grads <- Backward(meta_params = meta_params, y = y, models = models)
# meta_grads
