# This script prints summary of the model, like original Keras
#
# Created by Kunhong Yu(444447)
# Date: 2022/05/05
library(glue)

setGeneric(name = 'summary2',
           def = function(object, ...){
             standardGeneric('summary2')
           })
setMethod(f = 'summary2',
          signature = 'FullyModel',
          definition = function(object, models){
            # This function prints summary of the model
            # return :
            #   --models: models we obtain
            models2 <- models$model_ins
            summary_str <- ""
            # we go one forward with simple x
            input_shape <- models2[['hidden1']][['dense_layer']]@in_dim
            x <- matrix(rnorm(input_shape), nrow = 1, ncol = input_shape)
            meta_params <- Forward(models, x)

            summary_str <- paste0(summary_str,
                                  "----------------------------------------------------------------\n")
            line_new <- paste0(sprintf('%15s', 'Layer Name'), sprintf('%17s', 'Output Shape'),
                               sprintf('%17s', 'Params #\n'))
            # line_new <- paste0(str_replace_all(toString(rep(' ', 5)), ',', ''),
            #                    'Layer Name', str_replace_all(toString(rep(' ', 5)), ',', ''),
            #                    'Output Shape', str_replace_all(toString(rep(' ', 5)), ',', ''),
            #                    'Param #\n')
            summary_str <- paste0(summary_str, line_new)
            summary_str <- paste0(summary_str,
                                  "================================================================\n")

            # for each layer, we can print them
            total_params <- 0
            total_output <- 0
            meta_names <- names(meta_params)
            for (name in meta_names){
              x <- meta_params[[name]]$x
              dimx <- paste0('[None, ', dim(x)[2], ']')
              num_params <- length(c(meta_params[[name]]$W)) + length(c(meta_params[[name]]$b))
              line_new <- paste0(sprintf('%15s', name), sprintf('%15s', dimx),
                                 sprintf('%15d', num_params), '\n')
              # line_new <- paste0(str_replace_all(toString(rep(' ', 5)), ',', ''),
              #                    name, str_replace_all(toString(rep(' ', 5)), ',', ''),
              #                    dimx, str_replace_all(toString(rep(' ', 5)), ',', ''), num_params, '\n')
              summary_str <- paste0(summary_str, line_new)
              total_params <- total_params + num_params
              total_output <- total_output + dim(x)[2]
            }
            summary_str <- paste0(summary_str,
                                  "================================================================\n")
            line_new <- paste0('Total params: ', total_params, '\n')
            summary_str <- paste0(summary_str, line_new)

            summary_str <- paste0(summary_str,
                                  "----------------------------------------------------------------\n")
            total_output_size <- abs(2. * total_output * 4. /
                                    (1024 ** 2.))  # x2 for gradients
            total_params_size <- abs(total_params * 4. / (1024 ** 2.))
            total_size <- total_params_size + total_output_size

            summary_str <- paste0(summary_str, "Forward/backward pass size (MB): ", round(total_output_size, 2), '\n')
            summary_str <- paste0(summary_str, "Params size (MB): ", round(total_params_size, 2), "\n")
            summary_str <- paste0(summary_str, "Estimated Total Size (MB): ", round(total_size, 2), "\n")

            summary_str <- paste0(summary_str,
                                  "----------------------------------------------------------------\n")
            print(glue(summary_str))
          })

#' Summary of FullyLight model
#'
#' Keras-like summary of the model
#' @param models 'Model' instance, from \code{FullyLight::Model} function.
#' @export
Summary <- function(models){
  # This function prints summary of the model
  # return :
  #   --models: models we obtain
  summary_func <- selectMethod('summary2', signature = 'FullyModel')
  summary_func(models = models)
}

# unit test
# dims <- c(784, 1024, 2096, 1024, 10)
# models <- Model(dims = dims, input_shape = 4, hidden_activation = hidden_activation,
#                 out_activation = 'softmax', kernel_initializer = kernel_initializer, l1 = l1, l2 = l2)
# Summary(models)
