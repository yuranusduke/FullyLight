# This script defines classification metrics for our project
# Specifically, we define multiple metrics: acc, f1 in a macro way, recall in a macro way, precision in a macro way
# 
# Created by Kunhong Yu(444447)
# Date: 2022/05/04
setClass('Metrics',
         slots = list(method = 'character'),
         validity = function(object){
           if (!is.character(object@method)) return ('Method must be character, 
                                                     acc/f1/recall/precision are valid!')
           return (T)
         }
)

# 1. acc
setGeneric(name = 'acc',
           def = function(object, ...){
             standardGeneric('acc')
           })
setMethod(f = 'acc',
          signature = 'Metrics',
          definition = function(object, output, y){
            # Define acc metric
            # Args :
            #   --output: softmax output
            #   --y: one-hot encoding labels
            # return :
            #   --acc
            # 1. get preds
            preds <- apply(output, 1, function(x) which.max(x))
            # 2. get label
            labels <- apply(y, 1, function(x) which.max(x))
            
            acc <- round(sum(preds == labels) / length(labels), 5)
            
            return (acc)
          })

# 2. recall
setGeneric(name = 'recall',
           def = function(object, ...){
             standardGeneric('recall')
           })
setMethod(f = 'recall',
          signature = 'Metrics',
          definition = function(object, output, y){
            # Define recall metric
            # Args :
            #   --output: softmax output
            #   --y: one-hot encoding labels
            # return :
            #   --recall
            # 1. get preds
            preds <- apply(output, 1, function(x) which.max(x))
            # 2. get label
            labels <- apply(y, 1, function(x) which.max(x))
            
            # get rid of bug
            ctable_m <- table(factor(preds, levels = min(labels) : max(labels)), 
                              factor(labels, levels = min(labels) : max(labels)))
            recall <- round(mean(diag(ctable_m) / colSums(ctable_m), na.rm = T), 5)
            
            return (recall)
          })

# 3. precision
setGeneric(name = 'precision',
           def = function(object, ...){
             standardGeneric('precision')
           })
setMethod(f = 'precision',
          signature = 'Metrics',
          definition = function(object, output, y){
            # Define precision metric
            # Args :
            #   --output: softmax output
            #   --y: one-hot encoding labels
            # return :
            #   --recall
            # 1. get preds
            preds <- apply(output, 1, function(x) which.max(x))
            # 2. get label
            labels <- apply(y, 1, function(x) which.max(x))
            
            ctable_m <- table(factor(preds, levels = min(labels) : max(labels)), 
                              factor(labels, levels = min(labels) : max(labels)))
            precision <- round(mean(diag(ctable_m) / rowSums(ctable_m), na.rm = T), 5)
            
            return (precision)
          })

# 4. f1
setGeneric(name = 'f1',
           def = function(object, ...){
             standardGeneric('f1')
           })
setMethod(f = 'f1',
          signature = 'Metrics',
          definition = function(object, output, y){
            # Define f1 metric
            # Args :
            #   --output: softmax output
            #   --y: one-hot encoding labels
            # return :
            #   --recall
            # 1. get preds
            preds <- apply(output, 1, function(x) which.max(x))
            # 2. get label
            labels <- apply(y, 1, function(x) which.max(x))
            
            ctable_m <- table(factor(preds, levels = min(labels) : max(labels)), 
                              factor(labels, levels = min(labels) : max(labels)))
            precision <- mean(diag(ctable_m) / rowSums(ctable_m), na.rm = T)
            recall <- mean(diag(ctable_m) / colSums(ctable_m), na.rm = T) # micro
            f1 <- round(2. * precision * recall / (precision + recall), 5)
            
            return (f1)
          })

Metrics <- function(method = 'acc'){
  # Define Metrics functions
  # Args : 
  #   --method: 'acc'/'f1'/'recall'/'precision'
  # return : 
  #   --closure function FUNC
  stopifnot(method %in% c('acc', 'f1', 'recall', 'precision')) # defensive programming
  metrics <- new(Class = 'Metrics', method = method)
  FUNC <- function(output, y){
    # This inside function is used to compute metric function of input x
    # Args : 
    #   --output: softmax output
    #   --y: one-hot encoding label
    # return :
    #   --metric function
    if (method == 'acc') return (acc(metrics, output = output, y = y))
    else if (method == 'f1') return (f1(metrics, output = output, y = y))
    else if (method == 'precision') return (precision(metrics, output = output, y = y))
    else if (method == 'recall') return (recall(metrics, output = output, y = y))
  }
  return (FUNC) # closure
}

# unit test
# metric <- Metrics('acc')
# output <- matrix(c(1, 2, 4, 5), nrow = 2)
# output <- exp(output) / rowSums(exp(output)) # softmax
# label <- matrix(c(1, 0, 0, 1), nrow = 2, byrow = T)
# metric(output, label)

