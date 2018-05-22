//Author: Diego Vergara
#ifndef SOFTMAX_R
#define SOFTMAX_R
#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "../utils/c_utils.hpp"

using namespace Eigen;
using namespace std;


class SoftmaxRegression
{
 public:
 	SoftmaxRegression();
	void init(bool _normalization = false, bool _standardization = false,bool _with_bias=false);
	void init(MatrixXd &_X,VectorXd &_Y,double reg=1.0, bool _normalization = false, bool _standardization = true,bool _with_bias=true);
 	double logPosterior();
 	void setWeights(MatrixXd &_W);
    void setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing = true);
 	MatrixXd getWeights();
 	int getNClasses();
 	VectorXd getClasses();
 	void preCompute();
 	RowVectorXd featureMean,featureStd,featureMin,featureMax;
 	bool initialized = false;

 protected:
 	MatrixXd weights;
 	MatrixXd *X_train;
 	VectorXd *Y_train, classes;
	MatrixXd eta, phi, labelBinarize;
 	VectorXd momemtum;
 	int rows,dim, n_classes;
 	double reg;
 	bool normalization, standardization, with_bias;
 	//double logPrior();
 	MatrixXd logsoftmax(MatrixXd &eta);
 	MatrixXd softmax(MatrixXd &eta);
 	double logLikelihood();
 	MatrixXd Hessian;
 	C_utils tools;
    //MVNGaussian posterior;
};

#endif
