//Author: Diego Vergara
#ifndef CPU_LOGISTIC_H
#define CPU_LOGISTIC_H
#include "softmax_regression.hpp"

class CPU_SoftmaxRegression : public SoftmaxRegression
{
 public:
 	MatrixXd train(int n_iter, int mini_batch, double momentum=0.5,double learning_rate=0.1);
 	VectorXd predict(MatrixXd &_X_test, bool data_processing = true, bool _with_bias = true);
 	MatrixXd predict_proba(MatrixXd &_X_test, bool data_processing = true, bool _with_bias = true);
 	void preCompute(int iter,int mini_batch);
    MatrixXd computeGradient();
};

#endif
