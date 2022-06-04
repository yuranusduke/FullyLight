# This script is used for utilities in shiny app
# 
# Created by Kunhong Yu(444447)
# Date: 2022/05/20

validHyper <- function(dims, epochs, l1, l2){
  # This function is used to validate hyperparameters formats
  # Args :
  #   --all input fields in shiny app
  # return :
  #   --flag: all tests passed
  flag <- T
  if (is.na(as.numeric(l1))){
    shinyalert("Oops!", "L1 must be numeric!", type = "error")
    flag <- F
  } 
  if (is.na(as.numeric(l1))){
    shinyalert("Oops!", "L2 must be numeric!", type = "error")
    flag <- F
  }
  if (is.na(as.numeric(l1))){
    shinyalert("Oops!", "Epochs must be integer!", type = "error")
    flag <- F
  }
  if (length(dims) == 0) {
    flag <- F
  }
  else {
    dims <- str_split(dims, ',')[[1]]
    dims <- as.numeric(dims) # defensive programming
    #print(dims)
    if (any(is.na(dims))){
      shinyalert("Oops!", "Dims must be all integer with format of h1,h2,h3!", type = "error")
      flag <- F
    }
  }
  return (flag)
}