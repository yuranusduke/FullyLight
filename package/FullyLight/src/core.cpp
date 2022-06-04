// RCpp function is used to do core stuff in our engine
//
// Created by Kunhong Yu(444447)
#include <Rcpp.h>
using namespace Rcpp;

//' Compute crossentropy loss
//'
//' Loss function
//'
//' @param gt_y NumericMatrix. Element-wise multiplication of ground-truth label and softmax output
//' @return Calculated loss.
//' @export
// [[Rcpp::export]]
float CELoss(NumericMatrix gt_y) {
  // Compute loss
  // Args :
  //  --gt_y: gt labels with softmax output
  //  --output: softmax output
  // return :
  //  --loss
  float loss = mean(-log(rowSums(gt_y) + 1e-8)); //sugar functions
  
  return loss;
}
