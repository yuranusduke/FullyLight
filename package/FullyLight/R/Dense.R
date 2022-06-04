# This script defines Dense layer of NN
#
# Created by Kunhong Yu
# Date: 2022/05/03
# Step 0 Decide the structure of Net
# Our program must receive the units of neurons, activation function name and parameters initializer
# like: Dense(units = 10, activation = 'relu', kernel_initializer = 'random')
setClass('Dense',
         slots = list(in_dim = 'numeric',
                      out_dim = 'numeric',
                      activation = 'character',
                      kernel_initializer = 'character',
                      l1 = 'numeric',
                      l2 = 'numeric'),
         validity = function(object){
           if (!is.numeric(object@in_dim)) return ('Input dimension must be numeric!')
           if (!is.numeric(object@out_dim)) return ('Output dimension must be numeric!')
           if (!is.character(object@activation)) return ("Activation function must be character,
                                                          relu/sigmoid/tanh/prelu are valid!")
           if (!is.character(object@kernel_initializer)) return ("kernel initializer must be character,
                                                                   random/xavier/ones/zero are valid!")
           if (!is.numeric(object@l1)) return ('L1 parameter must be numeric!')
           if (!is.numeric(object@l2)) return ('L2 parameter must be numeric!')

           return (T)
         })

# 5.2 Let's build our model step by step
# 5.2.1 For each layer, we compute its pre-activation forward pass
setGeneric(name = 'layer_pre_activation_forward',
           def = function(object, ...){
             standardGeneric('layer_pre_activation_forward')
           })
setMethod(f = 'layer_pre_activation_forward',
          signature = 'Dense', # belongs to `Dense` class
          definition = function(object, W, b, x){
            # Define forward pass for only a single layer
            # Operations are done with matrix multiplication
            # return the result
            # Args :
            #   --x: input data, matrix from previous layer's output, which has shape [m, n_prev]
            #       n_prev is out_dim from previous layer, and in_dim for current layer
            #   --W/b
            # return :
            #   --z: current layer pre-activation output, shape: [m, out_dim]
            z <- sweep(x %*% t(W), 2, b, '+') # broadcasting, vectorization
            return (z)
          })
setGeneric(name = 'layer_pre_activation_backward',
           def = function(object, ...){
             standardGeneric('layer_pre_activation_backward')
           })
setMethod(f = 'layer_pre_activation_backward',
          signature = 'Dense', # belongs to `Dense` class
          definition = function(object, A_prev, dZ, W){
            # Define backward pass for only a single layer
            # Operations are done with matrix multiplication
            # return the result
            # Args :
            #   --A_prev: input data, matrix from previous layer's output, which has shape [m, n_prev]
            #       n_prev is out_dim from previous layer, and in_dim for current layer
            #   --dZ: upstream gradient, has shape [m, out_dim]
            #   --W
            # return :
            #   --dA_prev: previous layer's activation gradient
            #   --dW: previous layer's weights gradient
            #   --db: previous layer's bias gradient
            # All in Vectorization
            m <- dim(dZ)[1] # number of examples
            dA_prev <- dZ %*% W # previous layer's activation
            dW <- (1. / m) * (t(dZ) %*% A_prev) + 2. * object@l2 * W + object@l1 * sign(W)# W's gradient
            db <- colSums(dZ) # no regularization for bias
            db <- (1. / m) *  matrix(db, nrow = 1) # b's gradient

            return (list(dA_prev = dA_prev, dW = dW, db = db))
          })

# 5.2.2 Compute its activation forward pass
setGeneric(name = 'layer_activation_forward',
           def = function(object, ...){
             standardGeneric('layer_activation_forward')
           })
setMethod(f = 'layer_activation_forward',
          signature = 'Dense', # belongs to `Dense` class
          definition = function(object, x, W, b, activationFUNC){
            # Define forward pass for only a single layer
            # Operations are done with matrix multiplication
            # return the result
            # Args :
            #   --x: input data, matrix from previous layer's output, which has shape [m, n_prev]
            #       n_prev is out_dim from previous layer, and in_dim for current layer
            #   --W/b
            #   --activationFUNC: activation function
            # return :
            #   --x_prev: previous output
            #   --W: weights of current layer
            #   --b: bias of current layer
            #   --z: current layer pre-activation output, shape: [m, out_dim]
            #   --x: current forward pass result with activation
            # Initialie first
            x_prev <- x
            layer <- selectMethod('layer_pre_activation_forward', signature = "Dense")
            z <- layer(object, x = x, W = W, b = b)
            x <- activationFUNC(z)
            return (list(x_prev = x_prev, W = W, b = b, z = z, x = x))
          })
setGeneric(name = 'layer_activation_backward',
           def = function(object, ...){
             standardGeneric('layer_activation_backward')
           })
setMethod(f = 'layer_activation_backward',
          signature = 'Dense', # belongs to `Dense` class
          definition = function(object, x_prev, x_cur, dA, W,
                                activationBackwardFUNC, y = NULL){
            # Define backward pass for only a single layer with activation
            # Operations are done with matrix multiplication
            # return the result
            # Args :
            #   --x_prev: input data, matrix from previous layer's output, which has shape [m, n_prev]
            #       n_prev is out_dim from previous layer, and in_dim for current layer
            #   --x_cur: current layer output, shape : [m, out_dim]
            #   --dA: upstream gradient, has shape [m, out_dim]
            #   --W/b
            #   --activation: activation name
            #   --y
            # return :
            #   --dA_prev: previous layer's activation gradient
            #   --dW: previous layer's weights gradient
            #   --db: previous layer's bias gradient
            if (!is.null(y)) dZ <- dA * activationBackwardFUNC(x = x_cur, # for softmax
                                                               y = y) # element-wise multiplication
            else dZ <- dA * activationBackwardFUNC(x = x_cur) # element-wise multiplication
            backward_pre <- selectMethod('layer_pre_activation_backward', signature = "Dense")
            res <- backward_pre(object, A_prev = x_prev, dZ = dZ, W = W)
            dA_prev <- res$dA_prev
            dW <- res$dW
            db <- res$db
            return (list(dA = dA_prev, dW = dW, db = db))
          })

# Define constructor for class Dense
Dense <- function(in_dim, out_dim, activation = 'relu', kernel_initializer = 'xavier',
                  l1 = 0., l2 = 0.){
  # Define constructor for Dense class
  # Args :
  #   --in_dim: input dimension
  #   --out_dim: output dimension
  #   --activation: default is 'relu'
  #   --kernel_initializer: default is 'xavier'
  #   --l1/l2: parameter
  # return :
  #   --return closure function
  dense_layer <- new(Class = 'Dense', in_dim = in_dim, out_dim = out_dim, activation = activation,
                     kernel_initializer = kernel_initializer, l1 = l1, l2 = l2)
  # Before hand, we need to initialize
  params <- Initializer(method = kernel_initializer,
                        in_dim = in_dim, out_dim = out_dim)
  forward <- function(x, params2 = NULL){
    # This closure function is used to compute its forward pass for its layer
    # Args :
    #   --x: input data, from previous layer
    #   --params2
    # return :
    #   --forward: closure function
    if (!is.null(params2)) params <- params2
    ac <- Activation(method = activation)
    result <- layer_activation_forward(dense_layer, x = x, W = params$W, b = params$b,
                                       activationFUNC = ac$FUNC)
    result$ac_method <- ac$method # record which activation function used
    return (result)
  }

  return (list(forward = forward, dense_layer = dense_layer))
}

# Define Dense_Backward for class Dense
Dense_Backward <- function(dense_layer, x_prev, x_cur, dA, W, activation_method,
                           y = NULL){
  # Define Dense_Backward for Dense class
  # Args :
  #   --dense_layer: dense layer instance
  #   --x_prev: previous layer's activation
  #   --x_cur: current layer's activation
  #   --dA: current layer's activation gradients
  #   --W
  #   --activation_method: 'relu'/'prelu'/'sigmoid'/'tanh'/'linear'/'softmax'
  #   --y: one-hot encoding label
  # return :
  #   --res, containing dA, dW, db
  layer_backward <- selectMethod('layer_activation_backward', signature = 'Dense')
  activationBackwardFUNC <- Activation_Backward(method = activation_method)
  res <- layer_backward(dense_layer, x_prev = x_prev, dA = dA, y = y, x_cur = x_cur,
                        W = W,
                        activationBackwardFUNC = activationBackwardFUNC)

  return (res)
}

# unit test
# x <- matrix(c(2, 3, 4, 5), nrow = 2)
# func <- Dense(in_dim = 2, out_dim = 2, activation = 'relu',
#               kernel_initializer = 'xavier')
# temp <- func(x)
# temp
# #
# x_prev <- x
# dA <- matrix(c(1, 2, 3, 4), nrow = 2)
# x_cur <- x_prev
# activation_method <- 'relu'
# W <- matrix(rnorm(4), nrow = 2)
# res <- Dense_Backward(func$dense_layer, x_prev, x_cur, W = W, dA, activation_method)
# res
#
