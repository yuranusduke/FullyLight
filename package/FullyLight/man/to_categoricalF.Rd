% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{to_categoricalF}
\alias{to_categoricalF}
\title{Change y into categorical(one-hot encoding).}
\usage{
to_categoricalF(y, num_classes)
}
\arguments{
\item{y}{vector. Contains ground-truth labels.}

\item{num_classes}{numeric. Number of classes.}
}
\value{
One-hot encoding labels.
}
\description{
Keras-like to_categorical
}
\examples{
y <- c(1, 2, 3)
y_ <- to_categoricalF(y, 3)
}
