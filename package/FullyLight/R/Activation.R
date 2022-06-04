# This script defines Activation functions of NN
#
# Created by Kunhong Yu
# Date: 2022/05/03
# 5.1 Activation function definitions
setClass('Activation',
         slots = list(method = 'character'),
         validity = function(object){
           if (!is.character(object@method)) return ('Method must be character,
                                                     relu/sigmoid/tanh/prelu/linear/softmax are valid!')
           return (T)
           }
         )

# 5.1.1 ReLU
# x = max(x, 0)
setGeneric(name = 'relu',
           def = function(object, ...){
             standardGeneric('relu')
           })
setMethod(f = 'relu',
          signature = 'Activation',
          definition = function(object, x){
            # Define ReLU activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            x <- ifelse(x > 0., x, 0.) # relu
            return (x)
          })
setGeneric(name = 'relu_backward',
           def = function(object, ...){
             standardGeneric('relu_backward')
           })
setMethod(f = 'relu_backward',
          signature = 'Activation',
          definition = function(object, x){
            # Define ReLU activation function for backward, compute its gradient
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix from FORWARD PASS!!!
            x <- 1. * (x > 0.)
            return (x)
          })

# 5.1.2 sigmoid
# x = 1. / (1. + exp(-x))
setGeneric(name = 'sigmoid',
           def = function(object, ...){
             standardGeneric('sigmoid')
           })
setMethod(f = 'sigmoid',
          signature = 'Activation',
          definition = function(object, x){
            # Define Sigmoid activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            x <-  1. / (1. + exp(-x)) # sigmoid
            return (x)
          })
setGeneric(name = 'sigmoid_backward',
           def = function(object, ...){
             standardGeneric('sigmoid_backward')
           })
setMethod(f = 'sigmoid_backward',
          signature = 'Activation',
          definition = function(object, x){
            # Define Sigmoid activation function for backward pass
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix, from FORWARD PASS!!!
            sigmoid <- selectMethod('sigmoid', signature = 'Activation')
            x <- (1. - x) * x
            return (x)
          })

# 5.1.3 tanh
# x = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
setGeneric(name = 'tanh2', # tanh is pre-defined, we have to change our function's name
           def = function(object, ...){
             standardGeneric('tanh2')
           })
setMethod(f = 'tanh2',
          signature = 'Activation',
          definition = function(object, x){
            # Define Tanh activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            x <- (exp(x) - exp(-x)) / (exp(x) + exp(-x)) # tanh
            return (x)
          })
setGeneric(name = 'tanh2_backward',
           def = function(object, ...){
             standardGeneric('tanh2_backward')
           })
setMethod(f = 'tanh2_backward',
          signature = 'Activation',
          definition = function(object, x){
            # Define Tanh activation function for backward pass
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix, from FORWARD PASS!!!
            tanh2 <- selectMethod('tanh2', signature = 'Activation')
            x <- 1. - x ** 2
            return (x)
          })

# 5.1.4 prelu
# x = x if x > 0 else p * x, p is a small parameter, we set 0.1 in this project!
setGeneric(name = 'prelu',
           def = function(object, ...){
             standardGeneric('prelu')
           })
setMethod(f = 'prelu',
          signature = 'Activation',
          definition = function(object, x){
            # Define PReLU activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            x <- ifelse(x > 0., x, 0.1 * x) # prelu
            return (x)
          })
setGeneric(name = 'prelu_backward',
           def = function(object, ...){
             standardGeneric('prelu_backward')
           })
setMethod(f = 'prelu_backward',
          signature = 'Activation',
          definition = function(object, x){
            # Define PReLU activation function for backward pass
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix, from FORWARD PASS!!!
            x <- ifelse(x > 0., 1., 0.1)
            return (x)
          })

# 5.1.5 linear
# x = x
setGeneric(name = 'linear',
           def = function(object, ...){
             standardGeneric('linear')
           })
setMethod(f = 'linear',
          signature = 'Activation',
          definition = function(object, x){
            # Define linear activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            return (x) # liear
          })
setGeneric(name = 'linear_backward',
           def = function(object, ...){
             standardGeneric('linear_backward')
           })
setMethod(f = 'linear_backward',
          signature = 'Activation',
          definition = function(object, x){
            # Define linear activation function for backward pass
            # Operations are done element-wise
            # return the result
            row <- dim(x)[1]
            col <- dim(x)[2]
            x <- matrix(rep(1., row * col), nrow = row)
            return (x)
          })

# 5.1.6 softmax
# x = exp(x) / sum(exp(x))
setGeneric(name = 'softmax',
           def = function(object, ...){
             standardGeneric('softmax')
           })
setMethod(f = 'softmax',
          signature = 'Activation',
          definition = function(object, x){
            # Define softmax activation function
            # Operations are done element-wise
            # return the result
            # Args :
            #   --x: input data, matrix
            x <- 1e-5 + x
            x <- exp(x - max(x)) # stable version
            de <- rowSums(x)
            x <- x / de
            return (x) # linear
          })
setGeneric(name = 'softmax_backward',
           def = function(object, ...){
             standardGeneric('softmax_backward')
           })
setMethod(f = 'softmax_backward',
          signature = 'Activation',
          definition = function(object, x, y = NULL){
            # Define softmax activation function for backward pass
            # Operations are done element-wise
            # return the result
            # we only use softmax in output, so the loss is cross entropy
            # loss = -log(softmax(z)), we directly compute its gradient
            # Args :
            #   --x: input data, from FORWARD PASS!!! x is softmax output
            #   --y: is one ont-hot encoding
            if (!is.null(y)) # final layer
              x <- x - y # !!!
            else {
              # for saliency map
              dimx <- dim(x)
              dA <- matrix(rep(1., dimx[1] * dimx[2]), nrow = dimx[1]) # all 1
              x <- dA * y # get only correct output's gradient
            }
            return (x)
          })

# Finally, define its constructor
Activation <- function(method = 'relu'){
  # Define activation functions
  # Args :
  #   --method: 'relu'/'prelu'/'sigmoid'/'tanh'/'linear'/'softmax', default is 'relu'
  # return :
  #   --closure function FUNC and activation method
  stopifnot(method %in% c('relu', 'prelu', 'sigmoid', 'tanh', 'linear', 'softmax')) # defensive programming
  activation <- new(Class = 'Activation', method = method)
  FUNC <- function(x){
    # This inside function is used to compute activation function of input x
    # Args :
    #   --x: input data
    # return :
    #   --x: output
    if (method == 'relu') return (relu(activation, x = x))
    else if (method == 'sigmoid') return (sigmoid(activation, x = x))
    else if (method == 'prelu') return (prelu(activation, x = x))
    else if (method == 'tanh') return (tanh2(activation, x = x))
    else if (method == 'softmax') return (softmax(activation, x = x))
    else if (method == 'linear') return (linear(activation, x = x))
  }
  return (list(method = method, FUNC = FUNC)) # closure
}

# Finally, define its constructor
Activation_Backward <- function(method = 'relu'){
  # Define activation backward pass
  # Args :
  #   --method: 'relu'/'prelu'/'sigmoid'/'tanh'/'linear'/'softmax', default is 'relu'
  # return :
  #   --backward function
  stopifnot(method %in% c('relu', 'prelu', 'sigmoid', 'tanh', 'linear', 'softmax')) # defensive programming

  if (method == 'relu') backward_func <- selectMethod('relu_backward', signature = 'Activation')
  else if (method == 'sigmoid') backward_func <- selectMethod('sigmoid_backward', signature = 'Activation')
  else if (method == 'prelu') backward_func <- selectMethod('prelu_backward', signature = 'Activation')
  else if (method == 'tanh') backward_func <- selectMethod('tanh2_backward', signature = 'Activation')
  else if (method == 'softmax') backward_func <- selectMethod('softmax_backward', signature = 'Activation')
  else if (method == 'linear') backward_func <- selectMethod('linear_backward', signature = 'Activation')

  return (backward_func)
}

# unit test
# x <- matrix(c(10, 2, 4, 5, 19, 80), nrow = 2)
# ac <- Activation(method = 'softmax')
# ac$FUNC(x)

# dA_prev <- matrix(c(10, 2, -4, 0), nrow = 2)
# ac <- Activation_Backward(method = 'relu')
# ac(x = dA_prev)
