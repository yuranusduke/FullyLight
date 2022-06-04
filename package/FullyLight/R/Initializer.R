# This script defines Initializer of parameters of NN
# 
# Created by Kunhong Yu
# Date: 2022/05/03
# Step 4 Initialize parameters
setClass('Initializer',
          slots = list(method = 'character',
                       in_dim = 'numeric',
                       out_dim = 'numeric'),
          validity = function(object){
            if (!is.character(object@method)) return ('Method must be character, 
                                                      random/xavier/ones/zero are valid!')
            if (!is.numeric(object@in_dim)) return ('Input dimension must be numeric!')
            if (!is.numeric(object@out_dim)) return ('Output dimension must be numeric!')
            
            return (T)
            }
         )

# 4.1 Define random initializer
setGeneric(name = 'random_initializer',
           def = function(object, ...){
             standardGeneric('random_initializer')
           })
setMethod(f = 'random_initializer',
          signature = 'Initializer',
          definition = function(object){
            # Define random initializer from standard normal distribution
            # return a random matrix, which has shape [out_dim, in_dim]
            # and a vector, which has shape [1, out_dim] in a list format
            in_dim <- object@in_dim
            out_dim <- object@out_dim
            w_vector <- 0.2 * rnorm(in_dim * out_dim, mean = 0, sd = 1) # weights
            b_vector <- rep(0., out_dim)
            W <- matrix(w_vector, nrow = out_dim, ncol = in_dim)
            b <- matrix(b_vector, nrow = 1, ncol = out_dim)
            
            return (list(W = W, b = b))
          })

# 4.2 Define xavier initializer
setGeneric(name = 'xavier_initializer',
           def = function(object, ...){
             standardGeneric('xavier_initializer')
           })
setMethod(f = 'xavier_initializer',
          signature = 'Initializer',
          definition = function(object){
            # Define xavier initializer from standard normal distribution
            # tutorial: https://cs230.stanford.edu/section/4/
            # return a random matrix, which has shape [out_dim, in_dim]
            # and a vector, which has shape [1, out_dim] in a list format
            in_dim <- object@in_dim
            out_dim <- object@out_dim
            w_vector <- rnorm(in_dim * out_dim, mean = 0, sd = sqrt(1 / in_dim)) # weights
            b_vector <- rep(0., out_dim)
            W <- matrix(w_vector, nrow = out_dim, ncol = in_dim)
            b <- matrix(b_vector, nrow = 1, ncol = out_dim)
            
            return (list(W = W, b = b))
          })

# 4.3 Define ones initializer
setGeneric(name = 'ones_initializer',
           def = function(object, ...){
             standardGeneric('ones_initializer')
           })
setMethod(f = 'ones_initializer',
          signature = 'Initializer',
          definition = function(object){
            # Define ones initializer with all ones in weights
            # which has shape [out_dim, in_dim]
            # and a vector, which has shape [1, out_dim] in a list format
            in_dim <- object@in_dim
            out_dim <- object@out_dim
            w_vector <- rep(1., in_dim * out_dim)
            b_vector <- rep(0., out_dim)
            W <- matrix(w_vector, nrow = out_dim, ncol = in_dim)
            b <- matrix(b_vector, nrow = 1, ncol = out_dim)
            
            return (list(W = W, b = b))
          })

# 4.4 Define zeros initializer
setGeneric(name = 'zeros_initializer',
           def = function(object, ...){
             standardGeneric('zeros_initializer')
           })
setMethod(f = 'zeros_initializer',
          signature = 'Initializer',
          definition = function(object){
            # Define ones initializer with all zeros in weights
            # which has shape [out_dim, in_dim]
            # and a vector, which has shape [1, out_dim] in a list format
            in_dim <- object@in_dim
            out_dim <- object@out_dim
            w_vector <- rep(0., in_dim * out_dim)
            b_vector <- rep(0., out_dim)
            W <- matrix(w_vector, nrow = out_dim, ncol = in_dim)
            b <- matrix(b_vector, nrow = 1, ncol = out_dim)
            
            return (list(W = W, b = b))
          })

# Finally, define its constructor 
Initializer <- function(method, in_dim, out_dim){
  # Define kernel initializer
  # Args : 
  #   --method: 'ones'/'zeros'/'random'/'xavier'
  #   --in_dim: input dimension
  #   --out_dim: output dimension
  # return : 
  #   --initialized W and b
  stopifnot(method %in% c('random', 'xavier')) # defensive programming
  initializer <- new(Class = 'Initializer',
                     method = method, in_dim = in_dim,
                     out_dim = out_dim)
  if (method == 'random') return (random_initializer(initializer))
  else if (method == 'ones') return (ones_initializer(initializer))
  else if (method == 'zeros') return (zeros_initializer(initializer))
  else if (method == 'xavier') return (xavier_initializer(initializer))
}

# unit test
# params <- Initializer(method = 'random', in_dim = 2, out_dim = 10)
# params
