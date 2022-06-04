# This script builds Cross Entroy loss
# Because we only use CE loss for our project, but can be extended with more losses
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/04

setClass('Loss',
         slots = list(method = 'character',
                      l1 = 'numeric',
                      l2 = 'numeric'),
         validity = function(object){
           if (!is.character(object@method)) return ('Method must be character,
                                                     celoss are valid!')
           if (!is.numeric(object@l1)) return ('L1 parameter must be numeric!')
           if (!is.numeric(object@l2)) return ('L2 parameter must be numeric!')

           return (T)
         })

# loss function
setGeneric(name = 'celoss',
           def = function(object, ...){
             standardGeneric('celoss')
           })
## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @useDynLib FullyLight
setMethod(f = 'celoss',
          signature = 'Loss',

          definition = function(object, output, y, params = NULL){
            # Define cross entropy loss
            # Args :
            #   --output: softmax output, shape: [m, num_classes]
            #   --y: output one-hot encoding label, shape: [m, num_classes]
            #   --params: default is NULL
            # return :
            #   --celoss: celoss
            # celoss <- mean(-log(rowSums(y * output) + 1e-8))
            celoss <- CELoss(y * output)
            #print(log(rowSums(y * output)))
            if (!is.null(params)){ # used in training
              meta_names <- names(params)
              for (name in meta_names){
                celoss <- celoss + object@l2 * sum(params[[name]]$W ** 2) + object@l1 * sum(abs(params[[name]]$W))
              }
            }

            return (celoss)
          })

Loss <- function(method = 'celoss', l1 = 0.1, l2 = 0.1){
  # Define optimizer functions
  # Args :
  #   --method: 'celoss'
  #   --l1/l2
  # return :
  #   --closure function FUNC
  stopifnot(method %in% c('celoss')) # defensive programming
  loss <- new(Class = 'Loss', method = method, l1 = l1, l2 = l2)
  FUNC <- function(output, y, params = NULL, ...){
    # This inside function is used to compute optimizer function of input x
    # Args :
    #   --output: softmax output
    #   --y: one-hot encoding label
    #   --params: default is NULL
    # return :
    #   --loss function
    if (method == 'celoss') return (celoss(loss, output = output, y = y, params = params))
  }
  return (FUNC) # closure
}

# unit test
# loss <- Loss()
# output <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
# output <- exp(output) / rowSums(exp(output)) # softmax
# label <- matrix(c(1, 0, 0, 0, 1, 0), nrow = 2, byrow = T)
# loss(output, label)
