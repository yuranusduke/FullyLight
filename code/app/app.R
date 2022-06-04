# Script is used for visualizing using shiny
#
# Created by Kunhong Yu(444447)
# Date: 2022/06/02

my_path <- '/Users/kunhongyu/Desktop/DSBA/Year1/Semester2/Advanced R/project/code/'
setwd(my_path)
file.sources <- list.files(my_path,
                           pattern = "[^(app)|*].R$", full.names = TRUE,
                           ignore.case = TRUE, recursive = T)
sapply(file.sources, source)

library(assertthat)
library(shiny)
library(ggplot2)
library(dplyr)
library(shinythemes)
library(stringr)
library(shinyalert)
library(FullyLight)
library(caret)
#sourceCpp('./engine/core.cpp')

################# shiny ###############
ui <- shinyUI(fluidPage(theme = shinytheme("lumen"),
                        shinyjs::useShinyjs(),
                        tags$head(tags$style("#model{overflow-y:scroll;
                                              font-size:15px;
                                              max-height:160px}")),
  titlePanel("FullyLight: Keras-like Neural Network Classifier"),
  useShinyalert(),
  wellPanel(
    fluidRow(
    column(width = 12,
           align = 'center',
           column(width = 12,
                  align = 'center',
                  h3("Model Architecture & Visualization"),
                  column(width = 4, 
                         align = 'center',
                         div(id = 'vis', height = 20, width = 20)
                  ),
                  column(width = 4, #style = 'border-left: 1px solid',
                          align = 'center', offset = 0.5,
                          div(id = 'his', height = 20, width = 20)
                   ),
                  column(width = 4, #style = 'border-left: 1px solid',
                          align = 'center', offset = 0.5,
                          div(id = 'clas', height = 20, width = 20)
                   ),
    )
  ))),
  hr(),
  wellPanel(fluidRow(column(width = 12,
                  align = 'center',
                  h3("Training Log"),
                  column(width = 12, verbatimTextOutput("model")))
  )),
  hr(),
  wellPanel(fluidRow(
    column(
      width = 12,
      align = "center",
      h3("Hyper-parameters"),
      column(width = 6,
        column(width = 6, offset = 0.5, textInput(
               inputId = "dims",
               label = "Input Dimensions of Neural Network",
               value = "",
               width = "250px",
               placeholder = "h1,h2,h3(h3 is # of classes)")),
        column(width = 6, offset = 0.5, textInput(
               inputId = "epochs",
               label = "Input Training Epochs",
               value = "",
               width = "250px",
               placeholder = "integer value...")),
        column(width = 6, offset = 0.5, textInput(
               inputId = "l1",
               label = "Input L1 Regularization",
               value = "",
               width = "250px",
               placeholder = "numeric value...")),
        column(width = 6, offset = 0.5, textInput(
               inputId = "l2",
               label = "Input L2 Regularization",
               value = "",
               width = "250px",
               placeholder = "numeric value..."))
      ),
      column(width = 6, style = 'border-left: 1px solid',
        column(width = 6, offset = 0.5, selectInput(
          inputId = "data",
          label = "Choose Data Set",
          choices = c('MNIST(image/10)', 'FashionMNIST(image/10)',
                      'Iris(structured/3)', 'MTcars(structured/2)',
                      'Churn(structured/2)', 'Scat(structured/3)',
                      'Yeast(structured/10)',
                      'Random(2)'),
          selected = "MNIST(image)",
          multiple = F,
          selectize = FALSE,
          size = 1)),
        column(width = 6, offset = 0.5,selectInput(
               inputId = "initializer",
               label = "Choose Initializer",
               choices = c('Random', 'Xavier'),
               selected = "Xavier",
               multiple = F,
               selectize = FALSE,
               size = 1)),
        column(width = 6, offset = 0.5, selectInput(
               inputId = "activation",
               label = "Choose Activation Function",
               selected = "ReLU",
               choices = c('ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'Linear'),
               multiple = F,
               selectize = FALSE,
               size = 1)),
        column(width = 6, offset = 0.5, selectInput(
               inputId = "optimizer",
               label = "Choose Optimizer",
               choices = c('Adam', 'Momentum', 'SGD', 'RMSProp'),
               selected = "Adam",
               multiple = F,
               selectize = FALSE,
               size = 1)),
        column(width = 6, offset = 0.5,selectInput(
               inputId = "metric",
               label = "Choose Metric",
               selected = "Accuracy",
               choices = c('Accuracy', 'F1', 'Precision', 'Recall'),
               multiple = F,
               selectize = FALSE,
               size = 1)),
        column(width = 6, offset = 0.8, actionButton(
               inputId = "train",
               label = "Start training",
               #width = '120px',
               style = "color: #ffffff; background-color: #6d1f6b; border-color: #000000",
               #disabled = ""
            )),
      )
  )))
))

server <- function(input, output, session) {
  # Get hyperparameters
  dims <- reactive({input$dims})
  epochs <- reactive({input$epochs})
  l1 <- reactive({input$l1})
  l2 <- reactive({input$l2})
  dataset <- reactive({input$data})
  activation <- reactive({input$activation})
  metric <- reactive({input$metric})
  initializer <- reactive({input$initializer})
  optimizer <- reactive({input$optimizer})
  my_clicks <- reactiveValues(parameters = NULL)

  observeEvent(input$train, {
    file.sources <- list.files('./www',
                               pattern = "*.png$", full.names = TRUE,
                               ignore.case = TRUE, recursive = T)
    file.remove(file.sources)
    dims <- dims()
    epochs <- epochs()
    l1 <- l1()
    l2 <- l2()
    dataset <- dataset()
    activation <- activation()
    metric <- metric()
    initializer <- initializer()
    optimizer <- optimizer()

    if (epochs == '')
      epochs <- 20
    if (l1 == '')
      l1 <- 0.
    if (l2 == '')
      l2 <- 0.

    flag <- validHyper(dims = dims, epochs = epochs, l1 = l1, l2 = l2)
    if (!flag) my_clicks$parameters <- NULL
    else{
      my_clicks$parameters <- list(dims = dims, epochs = epochs, l1 = l1, l2 = l2,
                   dataset = dataset, activation = activation, metric = metric,
                   initializer = initializer, optimizer = optimizer)
      dims <- str_split(dims(), ',')[[1]]
      dims <- as.numeric(dims)
      withCallingHandlers({
          shinyjs::html("model", "")
          shinyjs::html("his", "")
          shinyjs::html("clas", "")
          shinyjs::html("vis", "")
          FL(data_name = dataset, dims = dims, hidden_activation = tolower(activation),
             kernel_initializer = tolower(initializer), optimizer = tolower(optimizer),
             metric = tolower(metric), epochs = as.numeric(epochs), l1 = as.numeric(l1),
             l2 = as.numeric(l2), output = T)
        },
        message = function(m) {
          shinyjs::html(id = "model", html = m$message, add = T) # update text
          # shinyjs::runjs('
          #   document.getElementById("model").scrollIntoView();
          # ') # auto scroll
          file.sourcesa <- list.files('./www',
                                     pattern = "*.png$", full.names = TRUE,
                                     ignore.case = TRUE, recursive = F)
          file.sources <- grep('clas', file.sourcesa, value = T)
          file.sources <- file.sources[order(as.numeric(gsub("[^0-9]+", "", file.sources)),
                                             decreasing = T)]
          pth <- paste(c("<img src='",
                  gsub('./www', '', file.sources[1]), "'height = 400 width = 400>"), collapse = '')
          shinyjs::html("clas", "")
          shinyjs::html(id = "clas", html = pth,
                        add = F) # update image for classification
          
          file.sources <- grep('his', file.sourcesa, value = T)
          file.sources <- file.sources[order(as.numeric(gsub("[^0-9]+", "", file.sources)),
                                             decreasing = T)]
          pth <- paste(c("<img src='",
                         gsub('./www', '', file.sources[1]), "'height = 400 width = 400>"), collapse = '')
          shinyjs::html("his", "")
          shinyjs::html(id = "his", html = pth,
                        add = F) # update image for metrics
          
          file.sources <- grep('plotNN', file.sourcesa, value = T)
          file.sources <- file.sources[order(as.numeric(gsub("[^0-9]+", "", file.sources)),
                                             decreasing = T)]
          pth <- paste(c("<img src='",
                         gsub('./www', '', file.sources[1]), "'height = 400 width = 400>"), collapse = '')
          shinyjs::html("vis", "")
          shinyjs::html(id = "vis", html = pth,
                        add = F) # update image for NN visualization
        })
    }
  })
}

shinyApp(ui, server)

