//Author: Diego Vergara
#include "CPU_softmax_regression.hpp"

MatrixXd CPU_SoftmaxRegression::train(int n_iter, double momentum, double learning_rate){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	MatrixXd velocity = MatrixXd::Zero(this->n_classes, this->dim);
	for(int i=0;i<n_iter;i++){
		//tools.printProgBar(i, n_iter);
		this->preCompute();
		log_likelihood(i)= this->logPosterior();
		if(i%1==0) cout << "iteration :   " << i << " | SUM loss : " << log_likelihood(i) << endl;
		MatrixXd gradient=this->computeGradient();
		velocity = momentum * velocity - (1.0 - momentum) * gradient;
		this->weights = this->weights + learning_rate * velocity;
	}
	return log_likelihood;
}

MatrixXd CPU_SoftmaxRegression::computeGradient(){
	//cout << this->eta << endl;
	this->phi = this->logsoftmax(this->eta);
	return -((this->labelBinarize.transpose()-this->phi.array().exp().matrix().transpose())* *this->X_train)  + this->reg*this->weights;
}

VectorXd CPU_SoftmaxRegression::predict(MatrixXd &_X_test, bool data_processing, bool _with_bias){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if(this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	if (this->with_bias and _with_bias){ //basis map: identity
		_X_test.conservativeResize(NoChange,this->dim);
		_X_test.col(this->dim-1) = VectorXd::Constant(_X_test.rows(), 1.0*0.01);
	}
	MatrixXd eta_test = (_X_test)*this->weights.transpose();
	MatrixXd phi_test=this->softmax(eta_test);
	VectorXd predict = tools.argMax(phi_test, true);
	for (int i = 0; i < predict.size(); ++i) predict(i) = this->classes(predict(i));
	return predict;
}

MatrixXd CPU_SoftmaxRegression::predict_proba(MatrixXd &_X_test, bool data_processing, bool _with_bias){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if(this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	if (this->with_bias and _with_bias){ //basis map: identity
		_X_test.conservativeResize(NoChange,this->dim);
		_X_test.col(this->dim-1) = VectorXd::Constant(_X_test.rows(), 1.0*0.1);
	}
	MatrixXd eta_test = (_X_test)*this->weights.transpose();
	MatrixXd phi_test=this->logsoftmax(eta_test);
	return phi_test;
}
