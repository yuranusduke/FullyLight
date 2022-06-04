# This script defines learning optimizers for our model
# 
# Created by Kunhong Yu(444447)
# Date: 2022/05/03
setClass('Optimizer',
         slots = list(method = 'character'),
         validity = function(object){
           if (!is.character(object@method)) return ('Method must be character, 
                                                     sgd/adam/rmsprop/momentum/ are valid!')
           return (T)
         }
)

# 1. sgd
setGeneric(name = 'sgd',
           def = function(object, ...){
             standardGeneric('sgd')
           })
setMethod(f = 'sgd',
          signature = 'Optimizer',
          definition = function(object, meta_grads, 
                                meta_params, 
                                learning_rate, ...){
            # Define SGD optimizer
            # Args :
            #   --meta_grads: obtained gradient 
            #   --meta_params: obtained parameters
            #   --learning_rate: learning rate
            # return :
            #   --meta_params: updated parameters
            meta_names <- names(meta_params)
            for (name in meta_names){
              meta_params[[name]]$W <- meta_params[[name]]$W - learning_rate * meta_grads[[name]]$dW
              meta_params[[name]]$b <- meta_params[[name]]$b - learning_rate * meta_grads[[name]]$db
            }
            
            return (list(meta_params = meta_params))
          })

# 2. momentum
setGeneric(name = 'momentum',
           def = function(object, ...){
             standardGeneric('momentum')
           })
setMethod(f = 'momentum',
          signature = 'Optimizer',
          definition = function(object, meta_grads, 
                                meta_params, 
                                beta = 0.9,
                                m = NULL,
                                learning_rate, ...){
            # Define momentum optimizer
            # Args :
            #   --meta_grads: obtained gradient 
            #   --meta_params: obtained parameters
            #   --beta: EMA parameter, default is 0.9
            #   --m: default is NULL to keep up with update
            #   --learning_rate: learning rate
            # return :
            #   --meta_params: updated parameters
            if (is.null(m)){
              m <- list()
              meta_names <- names(meta_params)
              for (name in meta_names){
                W_dim <- dim(meta_params[[name]]$W)
                m[[name]]$dW <- matrix(rep(0., W_dim[1] * W_dim[2]), nrow = W_dim[1]) # shape [out_dim, in_dim]
                m[[name]]$db <- matrix(rep(0., W_dim[1]), nrow = 1) # shape [1, out_dim]
              }
            }
            meta_names <- names(meta_params)
            for (name in meta_names){
              # update m
              m[[name]]$dW <- beta * m[[name]]$dW + (1. - beta) * meta_grads[[name]]$dW # momentum m
              m[[name]]$db <- beta * m[[name]]$db + (1. - beta) * meta_grads[[name]]$db
              # update parameters
              meta_params[[name]]$W <- meta_params[[name]]$W - learning_rate * m[[name]]$dW # momentum update
              meta_params[[name]]$b <- meta_params[[name]]$b - learning_rate * m[[name]]$db
            }
            
            return (list(meta_params = meta_params, m = m))
          })

# 3. rmsprop
setGeneric(name = 'rmsprop',
           def = function(object, ...){
             standardGeneric('rmsprop')
           })
setMethod(f = 'rmsprop',
          signature = 'Optimizer',
          definition = function(object, meta_grads, 
                                meta_params, 
                                beta = 0.999,
                                v = NULL,
                                learning_rate, ...){
            # Define rmsprop optimizer
            # Args :
            #   --meta_grads: obtained gradient 
            #   --meta_params: obtained parameters
            #   --beta: EMA parameter, default is 0.999
            #   --v: default is NULL to keep up with update
            #   --learning_rate: learning rate
            # return :
            #   --meta_params: updated parameters
            decay <- 1e-8
            if (is.null(v)){
              v <- list()
              meta_names <- names(meta_params)
              for (name in meta_names){
                W_dim <- dim(meta_params[[name]]$W)
                v[[name]]$dW <- matrix(rep(0., W_dim[1] * W_dim[2]), nrow = W_dim[1]) # shape [out_dim, in_dim]
                v[[name]]$db <- matrix(rep(0., W_dim[1]), nrow = 1) # shape [1, out_dim]
              }
            }
            meta_names <- names(meta_params)
            for (name in meta_names){
              # update v
              v[[name]]$dW <- beta * v[[name]]$dW + (1. - beta) * (meta_grads[[name]]$dW ** 2) # rmsprop v
              v[[name]]$db <- beta * v[[name]]$db + (1. - beta) * (meta_grads[[name]]$db ** 2)
              # update parameters
              meta_params[[name]]$W <- meta_params[[name]]$W - learning_rate * meta_grads[[name]]$dW / sqrt(v[[name]]$dW + decay) # rmsprop update
              meta_params[[name]]$b <- meta_params[[name]]$b - learning_rate * meta_grads[[name]]$db / sqrt(v[[name]]$db + decay)
            }
            
            return (list(meta_params = meta_params, v = v))
          })

# 4. adam
setGeneric(name = 'adam',
           def = function(object, ...){
             standardGeneric('adam')
           })
setMethod(f = 'adam',
          signature = 'Optimizer',
          definition = function(object, meta_grads, 
                                meta_params, 
                                beta1 = 0.9,
                                beta2 = 0.999,
                                m = NULL,
                                v = NULL,
                                learning_rate, ...){
            # Define adam optimizer, we omit bias correction!
            # Args :
            #   --meta_grads: obtained gradient 
            #   --meta_params: obtained parameters
            #   --beta: EMA parameter, default is 0.999
            #   --m: default is NULL to keep up with update for momentum
            #   --v: default is NULL to keep up with update for rmsprop
            #   --learning_rate: learning rate
            # return :
            #   --meta_params: updated parameters
            decay <- 1e-8
            if (is.null(v)){
              v <- list()
              m <- list()
              meta_names <- names(meta_params)
              for (name in meta_names){
                W_dim <- dim(meta_params[[name]]$W)
                m[[name]]$dW <- matrix(rep(0., W_dim[1] * W_dim[2]), nrow = W_dim[1]) # shape [out_dim, in_dim]
                m[[name]]$db <- matrix(rep(0., W_dim[1]), nrow = 1) # shape [1, out_dim]
                v[[name]]$dW <- matrix(rep(0., W_dim[1] * W_dim[2]), nrow = W_dim[1]) # shape [out_dim, in_dim]
                v[[name]]$db <- matrix(rep(0., W_dim[1]), nrow = 1) # shape [1, out_dim]
              }
            }
            meta_names <- names(meta_params)
            for (name in meta_names){
              # update m
              m[[name]]$dW <- beta1 * m[[name]]$dW + (1. - beta1) * meta_grads[[name]]$dW # momentum m
              m[[name]]$db <- beta1 * m[[name]]$db + (1. - beta1) * meta_grads[[name]]$db
              # update v
              v[[name]]$dW <- beta2 * v[[name]]$dW + (1. - beta2) * (meta_grads[[name]]$dW ** 2) # rmsprop v
              v[[name]]$db <- beta2 * v[[name]]$db + (1. - beta2) * (meta_grads[[name]]$db ** 2)
              # update parameters
              meta_params[[name]]$W <- meta_params[[name]]$W - learning_rate * m[[name]]$dW / sqrt(v[[name]]$dW + decay) # adam update
              meta_params[[name]]$b <- meta_params[[name]]$b - learning_rate * m[[name]]$db / sqrt(v[[name]]$db + decay)
            }
            
            return (list(meta_params = meta_params, v = v, m = m))
          })

Optimizer <- function(method = 'adam', learning_rate = 1e-3){
  # Define optimizer functions
  # Args : 
  #   --method: 'adam'/'sgd'/'momentum'/'rmsprop'
  #   --learning_rate: default is 1e-8
  # return : 
  #   --closure function FUNC
  stopifnot(method %in% c('adam', 'sgd', 'momentum', 'rmsprop')) # defensive programming
  optimizer <- new(Class = 'Optimizer', method = method)
  FUNC <- function(meta_grads, meta_params, ...){
    # This inside function is used to compute optimizer function of input x
    # Args : 
    #   --meta_grads
    #   --meta_params
    # return :
    #   --optimizer_func
    if (method == 'adam') return (adam(optimizer, meta_grads = meta_grads, meta_params = meta_params,
                                       learning_rate = learning_rate, ...))
    else if (method == 'sgd') return (sgd(optimizer, meta_grads = meta_grads, meta_params = meta_params,
                                          learning_rate = learning_rate, ...))
    else if (method == 'rmsprop') return (rmsprop(optimizer, meta_grads = meta_grads, meta_params = meta_params,
                                                learning_rate = learning_rate, ...))
    else if (method == 'momentum') return (momentum(optimizer, meta_grads = meta_grads, meta_params = meta_params,
                                                    learning_rate = learning_rate, ...))
  }
  return (FUNC) # closure
}

# unit test
# op <- Optimizer('adam')
# meta_grads <- list('hidden1' = list(dW = matrix(c(1, 2, 3, 4), nrow = 2), db = matrix(c(1, 1), nrow = 1)),
#                    'hidden2' = list(dW = matrix(c(3, 3, 3, 4), nrow = 2), db = matrix(c(1, 2), nrow = 1)))
# meta_params <- list('hidden1' = list(W = matrix(c(3, 2, -3, 4), nrow = 2), b = matrix(c(1, 0), nrow = 1)),
#                     'hidden2' = list(W = matrix(c(2, 3, -3, 4), nrow = 2), b = matrix(c(-6, 7), nrow = 1)))
# res <- op(meta_grads, meta_params, 1e-4)
# res
